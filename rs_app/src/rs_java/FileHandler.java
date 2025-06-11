package rs_java;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public class FileHandler {

    public static final int FILE_TIMEOUT_MS = 10 * 1000;

    public static final int SOURCE_SYMBOL_BYTE_SIZE = 2;
    public static final int SOURCE_SYMBOL_PER_PART = 1 << 15;
    public static final int TOTAL_SYMBOL_PER_PART = 1 << 16;
    public static final int SYMBOL_NEED_PER_PART = SOURCE_SYMBOL_PER_PART;
    public static final int PART_INIT_BYTE_SIZE = SOURCE_SYMBOL_BYTE_SIZE * SOURCE_SYMBOL_PER_PART;
    public static final int SYMBOL_PER_PACKET = 512;
    public static final int SEG_PER_PACKET = 256;
    public static final int SEG_DIFF = 1 << 15;
    public static final int PACKET_NEED_PER_PART = SYMBOL_NEED_PER_PART / SYMBOL_PER_PACKET;
    public static final int TOTAL_PACKET_PER_PART = TOTAL_SYMBOL_PER_PART / SYMBOL_PER_PACKET;

    private final RS_Code rsCode;
    private final FileHandlerCallback fileHandlerCallback;

    private final ConcurrentHashMap<UUID, FileEncodeData> encodeFile;
    private final ConcurrentHashMap<UUID, FileDecodeData> decodeFile;

    public FileHandler(FileHandlerCallback fileHandlerCallback) {

        this.rsCode = new RS_Code(this);
        this.fileHandlerCallback = fileHandlerCallback;

        encodeFile = new ConcurrentHashMap<>();
        decodeFile = new ConcurrentHashMap<>();

    }

    public void encodeProcessAfter(FileEncodeData fileEncodeData, long partId, int[] y) {
        // callback of encode part
        for (int i = 0; i < TOTAL_PACKET_PER_PART; i++) {
            byte packetId = (byte) i;
            Thread.ofVirtual().start(() -> {
                int[] symbols = new int[SYMBOL_PER_PACKET];
                int st = SEG_PER_PACKET * packetId;
                for (int j = 0; j < SEG_PER_PACKET; j++) {
                    symbols[j] = y[st + j];
                    symbols[j + SEG_PER_PACKET] = y[st + j + SEG_DIFF];
                }
                PacketHandler packetHandler = new PacketHandler(
                        fileEncodeData.getFileId(), fileEncodeData.getFileByteCnt(), partId, packetId, symbols);
                try {
                    packetHandler.genRawBytes();
                } catch (NoSuchAlgorithmException e) {
                    throw new RuntimeException(e);
                }
                fileHandlerCallback.encodePartProcessAfter(fileEncodeData, packetHandler);
            });
        }
    }

    public void sendPacketProcessAfter(FileEncodeData fileEncodeData, long partId) {
        if (fileEncodeData.getNumOfDoneEachPart()[(int) partId].incrementAndGet() == TOTAL_PACKET_PER_PART) {
            if (fileEncodeData.getNumOfDonePart().incrementAndGet() == fileEncodeData.getNumOfPart())
                fileHandlerCallback.encodeFileProcessAfter(fileEncodeData);
        }
    }

    public void decodeProcessAfter(FileDecodeData fileDecodeData, long partId, int[] p) {
        // callback of decode part
        long file_bytes_offset = partId * PART_INIT_BYTE_SIZE;
        genBytesFromDecodedSymbols(p, fileDecodeData.getFileBytes(), (int) file_bytes_offset);
        fileHandlerCallback.decodePartProcessAfter(fileDecodeData, partId);
    }

    public void mergePacketProcessAfter(FileDecodeData fileDecodeData, long partId) {
        fileDecodeData.getPartData()[(int) partId].setReceivedPacket(null);
        if (fileDecodeData.getNumOfDonePart().incrementAndGet() == fileDecodeData.getNumOfPart())
            fileHandlerCallback.decodeFileProcessAfter(fileDecodeData);
    }

    public void sendFileProcessAfter(FileEncodeData fileEncodeData) {
        Thread.ofVirtual().start(() -> {
            try {
                Thread.sleep(FILE_TIMEOUT_MS);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
            encodeFile.remove(fileEncodeData.getFileId());
        });
    }

    public void receiveFileProcessAfter(FileDecodeData fileDecodeData) {
//        Thread.ofVirtual().start(() -> {
//            try {
//                Thread.sleep(FILE_TIMEOUT_MS);
//            } catch (InterruptedException e) {
//                throw new RuntimeException(e);
//            }
//            decodeFile.remove(fileDecodeData.getFileId());
//        });
    }

    public FileEncodeData preEncode(byte[] data) {

        // call only in single thread, encode file
        UUID fileId = UUID.randomUUID();

        FileEncodeData fileEncodeData = new FileEncodeData(fileId, data.length);

        encodeFile.put(fileId, fileEncodeData);

        return fileEncodeData;
    }

    public FileDecodeData preDecode(PacketHandler packetHandler) throws NoSuchAlgorithmException {

        // call only in single thread, queue packet for decode
        FileDecodeData fileDecodeData = decodeFile.get(packetHandler.getFileId());
        if (fileDecodeData == null) {
            if (!packetHandler.verify()) return null;

            fileDecodeData = new FileDecodeData(packetHandler.getFileId(), packetHandler.getFileByteCnt());

            decodeFile.put(packetHandler.getFileId(), fileDecodeData);

            FileDecodeData fileDecodeDataForTrack = fileDecodeData;
            Thread.ofVirtual().start(() -> {
                while (true) {
                    try {
                        Thread.sleep(FILE_TIMEOUT_MS);
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                    if (fileDecodeDataForTrack.getNumOfDonePart().get() == fileDecodeDataForTrack.getNumOfPart())
                        break;
                    if (!fileDecodeDataForTrack.getTimeout().compareAndSet(false, true)) {
                        fileHandlerCallback.fileReceiveTimeoutProcessAfter(fileDecodeDataForTrack);
                        decodeFile.remove(fileDecodeDataForTrack.getFileId());
                        break;
                    }
                }
            });

        }

        return fileDecodeData;
    }

    public void encodeAsync(FileEncodeData fileEncodeData, byte[] data) {
        for (int i = 0; i < fileEncodeData.getNumOfPart(); i++) {
            long partId = i;
            Thread.ofVirtual().start(() -> {
                int[] symbols = new int[SOURCE_SYMBOL_PER_PART];
                long offset = PART_INIT_BYTE_SIZE * partId;
                genSourceSymbolsFromBytes(data, (int) offset, symbols);
                rsCode.encodeAsync(fileEncodeData, partId, symbols);
            });
        }
    }

    public void decodeAsync(FileDecodeData fileDecodeData, PacketHandler packetHandler) {
        Thread.ofVirtual().start(() -> {
            try {
                if (!packetHandler.verify()) {
                    return;
                }
            } catch (NoSuchAlgorithmException e) {
                throw new RuntimeException(e);
            }
            fileDecodeData.getTimeout().set(false);
            if (fileDecodeData.getNumOfDonePart().get() == fileDecodeData.getNumOfPart())
                return;

            PartDecodeData partDecodeData = fileDecodeData.getPartData()[(int) packetHandler.getPartId()];
            if (partDecodeData.getNumOfVerifyPacket().incrementAndGet() > PACKET_NEED_PER_PART)
                return;
            PacketHandler[] receivedPacket = partDecodeData.getReceivedPacket();
            receivedPacket[packetHandler.getId()] = packetHandler;
            if (partDecodeData.getNumOfReceivedPacket().incrementAndGet() == PACKET_NEED_PER_PART) {
                int[] x = new int[SYMBOL_NEED_PER_PART];
                int[] y = new int[SYMBOL_NEED_PER_PART];
                int st = 0;
                for (int k = 0; k < TOTAL_PACKET_PER_PART; k++) {
                    if (receivedPacket[k] == null) continue;
                    int st_x = k * SEG_PER_PACKET;
                    System.arraycopy(receivedPacket[k].getSymbols(), 0, y, st, SYMBOL_PER_PACKET);
                    for (int j = 0; j < SEG_PER_PACKET; j++) {
                        x[st + j] = st_x + j;
                        x[st + j + SEG_PER_PACKET] = st_x + j + SEG_DIFF;
                    }
                    st += SYMBOL_PER_PACKET;
                }
                rsCode.decodeAsync(fileDecodeData, packetHandler.getPartId(), x, y);
            }
        });
    }

    public static long calNumOfPart(long fileByteCnt) {
        return (fileByteCnt + PART_INIT_BYTE_SIZE - 1) / PART_INIT_BYTE_SIZE;
    }

    public static String getFileName(String path) {
        return Paths.get(path).getFileName().toString();
    }

    public static byte[] readFileBytes(String path) throws IOException {
        BufferedInputStream in = new BufferedInputStream(new FileInputStream(path), PART_INIT_BYTE_SIZE);
        return in.readAllBytes();
    }

    public static void writeFileBytes(String path, byte[] data) throws IOException {
        BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(path), PART_INIT_BYTE_SIZE);
        out.write(data);
        out.close();
    }

    public void finish() {
        if (rsCode != null)
            rsCode.finish();
    }

    public static void genBytesFromDecodedSymbols(int[] symbols, byte[] data, int offset) {
        ByteBuffer buffer = ByteBuffer.wrap(data, offset, FileHandler.PART_INIT_BYTE_SIZE);
        for (int i = 0; i < FileHandler.SOURCE_SYMBOL_PER_PART; i++) {
            buffer.put((byte) ((symbols[i] >> 8) & 0xFF));
            buffer.put((byte) (symbols[i] & 0xFF));
        }
    }

    public static void genSourceSymbolsFromBytes(byte[] data, int offset, int[] symbols) {
        ByteBuffer buffer = null;
        if (offset + FileHandler.PART_INIT_BYTE_SIZE <= data.length) {
            buffer = ByteBuffer.wrap(data, offset, FileHandler.PART_INIT_BYTE_SIZE);
        } else {
            // padding
            byte[] tmp = new byte[FileHandler.PART_INIT_BYTE_SIZE];
            System.arraycopy(data, offset, tmp, 0, data.length - offset);
            buffer = ByteBuffer.wrap(tmp);
        }
        for (int i = 0; i < FileHandler.SOURCE_SYMBOL_PER_PART; i++) {
            int symbol = ((buffer.get() & 0xFF) << 8) | (buffer.get() & 0xFF);
            symbols[i] = symbol;
        }
    }

    public static byte[] genChecksumMD5(byte[] data, int offset, int length) throws NoSuchAlgorithmException {
        MessageDigest md = MessageDigest.getInstance("MD5");
        md.update(data, offset, length);
        return md.digest();
    }

    public class FileEncodeData {
        private UUID fileId;
        private long fileByteCnt;
        private long numOfPart;
        private AtomicInteger numOfDonePart;
        private AtomicInteger[] numOfDoneEachPart;

        public FileEncodeData(UUID fileId, long fileByteCnt) {
            this.fileId = fileId;
            this.fileByteCnt = fileByteCnt;
            this.numOfPart = calNumOfPart(fileByteCnt);
            numOfDonePart = new AtomicInteger(0);
            numOfDoneEachPart = new AtomicInteger[(int) numOfPart];
            for (int i = 0; i < numOfPart; i++)
                numOfDoneEachPart[i] = new AtomicInteger(0);
        }

        public UUID getFileId() {
            return fileId;
        }

        public void setFileId(UUID fileId) {
            this.fileId = fileId;
        }

        public long getFileByteCnt() {
            return fileByteCnt;
        }

        public void setFileByteCnt(long fileByteCnt) {
            this.fileByteCnt = fileByteCnt;
        }

        public long getNumOfPart() {
            return numOfPart;
        }

        public void setNumOfPart(long numOfPart) {
            this.numOfPart = numOfPart;
        }

        public AtomicInteger getNumOfDonePart() {
            return numOfDonePart;
        }

        public void setNumOfDonePart(AtomicInteger numOfDonePart) {
            this.numOfDonePart = numOfDonePart;
        }

        public AtomicInteger[] getNumOfDoneEachPart() {
            return numOfDoneEachPart;
        }

        public void setNumOfDoneEachPart(AtomicInteger[] numOfDoneEachPart) {
            this.numOfDoneEachPart = numOfDoneEachPart;
        }
    }


    public class FileDecodeData {
        UUID fileId;
        long fileByteCnt;
        long numOfPart;
        AtomicInteger numOfDonePart;
        PartDecodeData[] partData;
        byte[] fileBytes;

        AtomicBoolean timeout;

        public FileDecodeData(UUID fileId, long fileByteCnt) {
            this.fileId = fileId;
            this.fileByteCnt = fileByteCnt;
            this.numOfPart = calNumOfPart(fileByteCnt);
            numOfDonePart = new AtomicInteger(0);
            partData = new PartDecodeData[(int) numOfPart];
            fileBytes = new byte[(int) numOfPart * PART_INIT_BYTE_SIZE];

            for (int i = 0; i < numOfPart; i++)
                partData[i] = new PartDecodeData();

            timeout = new AtomicBoolean(false);
        }

        public UUID getFileId() {
            return fileId;
        }

        public void setFileId(UUID fileId) {
            this.fileId = fileId;
        }

        public long getFileByteCnt() {
            return fileByteCnt;
        }

        public void setFileByteCnt(long fileByteCnt) {
            this.fileByteCnt = fileByteCnt;
        }

        public long getNumOfPart() {
            return numOfPart;
        }

        public void setNumOfPart(long numOfPart) {
            this.numOfPart = numOfPart;
        }

        public AtomicInteger getNumOfDonePart() {
            return numOfDonePart;
        }

        public void setNumOfDonePart(AtomicInteger numOfDonePart) {
            this.numOfDonePart = numOfDonePart;
        }

        public PartDecodeData[] getPartData() {
            return partData;
        }

        public void setPartData(PartDecodeData[] partData) {
            this.partData = partData;
        }

        public byte[] getFileBytes() {
            return fileBytes;
        }

        public void setFileBytes(byte[] fileBytes) {
            this.fileBytes = fileBytes;
        }

        public AtomicBoolean getTimeout() {
            return timeout;
        }

        public void setTimeout(AtomicBoolean timeout) {
            this.timeout = timeout;
        }
    }

    public class PartDecodeData {
        AtomicInteger numOfVerifyPacket;
        AtomicInteger numOfReceivedPacket;
        PacketHandler[] receivedPacket;

        public PartDecodeData() {
            numOfVerifyPacket = new AtomicInteger(0);
            numOfReceivedPacket = new AtomicInteger(0);
            receivedPacket = new PacketHandler[TOTAL_PACKET_PER_PART];
        }

        public AtomicInteger getNumOfVerifyPacket() {
            return numOfVerifyPacket;
        }

        public void setNumOfVerifyPacket(AtomicInteger numOfVerifyPacket) {
            this.numOfVerifyPacket = numOfVerifyPacket;
        }

        public AtomicInteger getNumOfReceivedPacket() {
            return numOfReceivedPacket;
        }

        public void setNumOfReceivedPacket(AtomicInteger numOfReceivedPacket) {
            this.numOfReceivedPacket = numOfReceivedPacket;
        }

        public PacketHandler[] getReceivedPacket() {
            return receivedPacket;
        }

        public void setReceivedPacket(PacketHandler[] receivedPacket) {
            this.receivedPacket = receivedPacket;
        }
    }

}

package rs_java;

import java.nio.ByteBuffer;
import java.security.NoSuchAlgorithmException;
import java.util.UUID;

public class PacketHandler {

    // packet structure
    public static final int FILE_ID_START = 0;
    public static final int FILE_ID_BYTE_SIZE = 16;
    public static final int FILE_BYTE_CNT_START = FILE_ID_START + FILE_ID_BYTE_SIZE;
    public static final int FILE_BYTE_CNT_BYTE_SIZE = 8;
    public static final int PART_ID_START = FILE_BYTE_CNT_START + FILE_BYTE_CNT_BYTE_SIZE;
    public static final int PART_ID_BYTE_SIZE = 8;
    public static final int ID_START = PART_ID_START + PART_ID_BYTE_SIZE;
    public static final int ID_BYTE_SIZE = 1;
    public static final int DATA_START = ID_START + ID_BYTE_SIZE;
    public static final int DATA_BYTE_SIZE = 1024;
    public static final int SUB_DATA_START = DATA_START + DATA_BYTE_SIZE;
    public static final int SUB_DATA_BYTE_SIZE = 64;
    public static final int CHECKSUM_START = SUB_DATA_START + SUB_DATA_BYTE_SIZE;
    public static final int CHECKSUM_BYTE_SIZE = 16;

    public static final int TOTAL_PACKET_SIZE = CHECKSUM_START + CHECKSUM_BYTE_SIZE;

    private UUID fileId;
    private long fileByteCnt;
    private long partId;
    private byte id;

    private byte[] rawBytes;

    private int[] symbols;

    public PacketHandler(byte[] rawBytes) {
        this.rawBytes = rawBytes;
    }

    public PacketHandler(UUID fileId, long fileByteCnt, long partId, byte id, int[] symbols) {
        this.fileId = fileId;
        this.fileByteCnt = fileByteCnt;
        this.partId = partId;
        this.id = id;
        this.symbols = symbols;
    }

    public void genRawBytes() throws NoSuchAlgorithmException {

        rawBytes = new byte[TOTAL_PACKET_SIZE];

        ByteBuffer buffer = ByteBuffer.wrap(rawBytes, 0, DATA_START);
        buffer.putLong(fileId.getMostSignificantBits());
        buffer.putLong(fileId.getLeastSignificantBits());
        buffer.putLong(fileByteCnt);
        buffer.putLong(partId);
        buffer.put(id);

        buffer = ByteBuffer.wrap(rawBytes, DATA_START, DATA_BYTE_SIZE);
        for (int i = 0; i < FileHandler.SYMBOL_PER_PACKET; i++) {
            buffer.put((byte) ((symbols[i] >> 8) & 0xFF));
            buffer.put((byte) (symbols[i] & 0xFF));
            rawBytes[SUB_DATA_START + (i >> 3)] |= (byte) (((symbols[i] >> 16) << (i & 7)) & 0xFF);
        }

        buffer = ByteBuffer.wrap(rawBytes, CHECKSUM_START, CHECKSUM_BYTE_SIZE);
        buffer.put(FileHandler.genChecksumMD5(rawBytes, 0, CHECKSUM_START));

    }

    public boolean verify() throws NoSuchAlgorithmException {
        byte[] checksum = FileHandler.genChecksumMD5(rawBytes, 0, CHECKSUM_START);
        for (int i = 0; i < CHECKSUM_BYTE_SIZE; i++)
            if (rawBytes[CHECKSUM_START + i] != checksum[i])
                return false;

        ByteBuffer buffer = ByteBuffer.wrap(rawBytes, 0, SUB_DATA_START);
        fileId = new UUID(buffer.getLong(), buffer.getLong());
        fileByteCnt = buffer.getLong();
        partId = buffer.getLong();
        id = buffer.get();

        symbols = new int[FileHandler.SYMBOL_PER_PACKET];
        for (int i = 0; i < FileHandler.SYMBOL_PER_PACKET; i++) {
            symbols[i] = (((buffer.get() & 0xFF) << 8) | (buffer.get() & 0xFF));
            symbols[i] |= (((rawBytes[SUB_DATA_START + (i >> 3)] >> (i & 7)) & 1) << 16);
        }

        return true;
    }

    public UUID getFileId() {
        if (fileId == null) {
            // not verify yet
            ByteBuffer buffer = ByteBuffer.wrap(rawBytes, FILE_ID_START, FILE_ID_BYTE_SIZE);
            return new UUID(buffer.getLong(), buffer.getLong());
        }
        return fileId;
    }

    public long getFileByteCnt() {
        return fileByteCnt;
    }

    public long getPartId() {
        if (fileId == null) {
            // not verify yet
            return ByteBuffer.wrap(rawBytes, PART_ID_START, PART_ID_BYTE_SIZE).getLong();
        }
        return partId;
    }

    public byte getId() {
        if (fileId == null) {
            // not verify yet
            return ByteBuffer.wrap(rawBytes, ID_START, ID_BYTE_SIZE).get();
        }
        return id;
    }

    public byte[] getRawBytes() throws NoSuchAlgorithmException {
        if (rawBytes == null) {}
            genRawBytes();
        return rawBytes;
    }

    public int[] getSymbols() {
        return symbols;
    }

}

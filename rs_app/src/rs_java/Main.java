package rs_java;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Scanner;
import java.util.concurrent.atomic.AtomicBoolean;

public class Main implements FileHandlerCallback {

    private static final boolean DEBUG = false;

    private static final int MAX_FILE_NAME_BYTE_SIZE = 127;
    private static final int FILE_NAME_BYTE_SIZE = 128;
    private static final int CHECKSUM_BYTE_SIZE = 16;

    private FileHandler fileHandler;
    private final String FILE_PATH;
    private DatagramSocket sendSocket, receiveSocket;
    private InetAddress dstIP;
    private int dstPort;

    private AtomicBoolean isRunning;

    public Main(int receivePort, int sendPort) throws IOException {

        sendSocket = new DatagramSocket(sendPort);
        receiveSocket = new DatagramSocket(receivePort);

        FILE_PATH = Paths.get("").toAbsolutePath() +
                FileSystems.getDefault().getSeparator() + "received" + FileSystems.getDefault().getSeparator();
        Files.createDirectories(Paths.get(FILE_PATH));

        isRunning = new AtomicBoolean(false);

    }

    public static void main(String[] args) {

        Scanner scanner = new Scanner(System.in);

        System.out.print("Send port: ");
        int sendPort = scanner.nextInt();
        System.out.print("Receive port: ");
        int receivePort = scanner.nextInt();

        Main main = null;
        try {
            main = new Main(receivePort, sendPort);
            System.out.println("OK.");
        } catch (Exception e) {
            if (DEBUG) e.printStackTrace();
            System.out.println("Setup fail, quit.");
            return;
        }

        // setup command is not thread safe, don't mix command
        System.out.println("Save path: " + main.FILE_PATH);
        System.out.println("Setup: -s {destination_ip} {destination_port}");
        System.out.println("Send: -f {file_path}");
        System.out.println("Quit: -q");
        scanner.nextLine();

        main.fileHandler = new FileHandler(main);

        // console app..
        main.isRunning.set(true);

        Thread receiveThread = new Thread(new ReceiveApp(main));
        receiveThread.start();

        while (main.isRunning.get()) {

            String input = scanner.nextLine().trim();
            if (input.equals("-q")) {
                main.isRunning.set(false);
                System.out.println("Bye.");
                continue;
            }

            String[] tokens = input.split("\\s+");
            if (tokens[0].equals("-f")) {

                if (main.dstIP == null) {
                    System.out.println("Need setup first, skip.");
                    continue;
                }

                String filePath = tokens[1];

                String fileName = FileHandler.getFileName(filePath);
                byte[] fileNameBytes = new byte[FILE_NAME_BYTE_SIZE];
                byte[] tmp = fileName.getBytes();
                if (tmp.length > MAX_FILE_NAME_BYTE_SIZE) {
                    System.out.println("File name too long (max " + MAX_FILE_NAME_BYTE_SIZE + " bytes), skip.");
                    continue;
                }
                System.arraycopy(tmp, 0, fileNameBytes, 0, tmp.length);
                byte paddingBytes = 0;
                if (tmp[tmp.length - 1] == paddingBytes) paddingBytes++;
                for (int i = tmp.length; i < FILE_NAME_BYTE_SIZE; i++)
                    fileNameBytes[i] = paddingBytes;

                byte[] fileBytes;
                try {
                    fileBytes = FileHandler.readFileBytes(filePath);
                } catch (Exception e) {
                    System.out.println("Read file error, skip.");
                    continue;
                }

                byte[] fullFileBytes = new byte[FILE_NAME_BYTE_SIZE + fileBytes.length + CHECKSUM_BYTE_SIZE];
                System.arraycopy(fileNameBytes, 0, fullFileBytes, 0, FILE_NAME_BYTE_SIZE);
                System.arraycopy(fileBytes, 0, fullFileBytes, FILE_NAME_BYTE_SIZE, fileBytes.length);
                try {
                    tmp = FileHandler.genChecksumMD5(fullFileBytes, 0, FILE_NAME_BYTE_SIZE + fileBytes.length);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
                System.arraycopy(tmp, 0, fullFileBytes, FILE_NAME_BYTE_SIZE + fileBytes.length, tmp.length);

                FileHandler.FileEncodeData fileEncodeData = main.fileHandler.preEncode(fullFileBytes);
                System.out.println("File " + fileName + " will be send as id " + fileEncodeData.getFileId());
                main.fileHandler.encodeAsync(fileEncodeData, fullFileBytes);

            } else if (tokens[0].equals("-s")) {

                InetAddress address;
                Integer port;

                try {
                    address = InetAddress.getByName(tokens[1]);
                    port = Integer.parseInt(tokens[2]);
                } catch (Exception e) {
                    System.out.println("Invalid ip or port, skip.");
                    continue;
                }

                main.dstIP = address;
                main.dstPort = port;

                System.out.println("OK.");

            } else {
                System.out.println("Invalid command, skip.");
            }


        }

        try {
            receiveThread.interrupt();
        } catch (Exception e) {
        }

        main.sendSocket.close();
        main.receiveSocket.close();
        main.fileHandler.finish();

    }

    public static class ReceiveApp implements Runnable {

        private Main main;

        public ReceiveApp(Main main) {
            this.main = main;
        }

        @Override
        public void run() {

            while (main.isRunning.get()) {

                byte[] buffer = new byte[PacketHandler.TOTAL_PACKET_SIZE];
                DatagramPacket packet = new DatagramPacket(buffer, buffer.length);
                try {
                    main.receiveSocket.receive(packet);
                } catch (Exception e) {
                    continue;
                }

                PacketHandler packetHandler = new PacketHandler(buffer);

                FileHandler.FileDecodeData fileDecodeData = null;
                try {
                    fileDecodeData = main.fileHandler.preDecode(packetHandler);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }

                if (fileDecodeData == null) {
                    if (DEBUG) System.out.println("Packet error..");
                    continue;
                }

                main.fileHandler.decodeAsync(fileDecodeData, packetHandler);

            }

        }

    }

    @Override
    public void encodePartProcessAfter(FileHandler.FileEncodeData fileEncodeData, PacketHandler packetHandler) {
        byte[] buffer = null;
        try {
            buffer = packetHandler.getRawBytes();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        DatagramPacket packet = new DatagramPacket(buffer, buffer.length, dstIP, dstPort);
        try {
            sendSocket.send(packet);
        } catch (Exception e) {
            return;
        }

        fileHandler.sendPacketProcessAfter(fileEncodeData, packetHandler.getPartId());

    }

    @Override
    public void encodeFileProcessAfter(FileHandler.FileEncodeData fileEncodeData) {
        System.out.println("File " + fileEncodeData.getFileId() + " send completed.");
        fileHandler.sendFileProcessAfter(fileEncodeData);
    }

    @Override
    public void decodePartProcessAfter(FileHandler.FileDecodeData fileDecodeData, long partId) {
//        if (DEBUG) System.out.println("Merge part " + partId + " of file " + fileDecodeData.getFileId());
        fileHandler.mergePacketProcessAfter(fileDecodeData, partId);
    }

    @Override
    public void decodeFileProcessAfter(FileHandler.FileDecodeData fileDecodeData) {

        byte[] fullFileBytes = new byte[(int) fileDecodeData.getFileByteCnt()];
        System.arraycopy(fileDecodeData.getFileBytes(), 0, fullFileBytes, 0, fullFileBytes.length);
        int dataLen = fullFileBytes.length - CHECKSUM_BYTE_SIZE;
        byte[] checksum = null;
        try {
            checksum = FileHandler.genChecksumMD5(fullFileBytes, 0, dataLen);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        for (int i = 0; i < CHECKSUM_BYTE_SIZE; i++) {
            if (checksum[i] != fullFileBytes[dataLen + i]) {
                System.out.println("File " + fileDecodeData.getFileId() + " receive fail.");
                return;
            }
        }

        byte[] tmp = new byte[FILE_NAME_BYTE_SIZE];
        System.arraycopy(fullFileBytes, 0, tmp, 0, FILE_NAME_BYTE_SIZE);
        int fileNameLen = FILE_NAME_BYTE_SIZE - 1;
        while (tmp[fileNameLen - 1] == tmp[fileNameLen]) {
            fileNameLen--;
        }
        String fileName = new String(tmp, 0, fileNameLen);
        System.out.println("File " + fileDecodeData.getFileId() + " original name is " + fileName);

        String filePath = FILE_PATH + fileDecodeData.getFileId().toString() + "_" + fileName;

        int fileDataLen = fullFileBytes.length - CHECKSUM_BYTE_SIZE - FILE_NAME_BYTE_SIZE;
        byte[] fileBytes = new byte[fileDataLen];
        System.arraycopy(fullFileBytes, FILE_NAME_BYTE_SIZE, fileBytes, 0, fileDataLen);
        try {
            FileHandler.writeFileBytes(filePath, fileBytes);
        } catch (Exception e) {
            System.out.println("File " + fileDecodeData.getFileId() + " write to disk fail.");
            return;
        }

        System.out.println("File " + fileDecodeData.getFileId() + " receive completed.");
        fileHandler.receiveFileProcessAfter(fileDecodeData);
    }

    @Override
    public void fileReceiveTimeoutProcessAfter(FileHandler.FileDecodeData fileDecodeData) {
        System.out.println("File " + fileDecodeData.getFileId() + " receive timeout, " +
                fileDecodeData.getNumOfDonePart().get() + "/" +
                fileDecodeData.getNumOfPart() + " parts, skip.");
    }

}
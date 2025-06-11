package rs_java;

public class Main implements FileHandlerCallback {


    public static void main(String[] args) {

        Main main = new Main();
        try (FileHandler fileHandler = new FileHandler(main)) {

            // ..

        }

    }


    @Override
    public void encodePartProcessAfter(FileHandler.FileEncodeData fileEncodeData, PacketHandler packetHandler) {

    }

    @Override
    public void encodeFileProcessAfter(FileHandler.FileEncodeData fileEncodeData) {

    }

    @Override
    public void decodePartProcessAfter(FileHandler.FileDecodeData fileDecodeData) {

    }

    @Override
    public void decodeFileProcessAfter(FileHandler.FileDecodeData fileDecodeData) {

    }
}
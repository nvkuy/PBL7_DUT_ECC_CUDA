package rs_java;

public interface FileHandlerCallback {

    void encodePartProcessAfter(FileHandler.FileEncodeData fileEncodeData, PacketHandler packetHandler);

    void encodeFileProcessAfter(FileHandler.FileEncodeData fileEncodeData);

    void decodePartProcessAfter(FileHandler.FileDecodeData fileDecodeData, long partId);

    void decodeFileProcessAfter(FileHandler.FileDecodeData fileDecodeData);

    void fileReceiveTimeoutProcessAfter(FileHandler.FileDecodeData fileDecodeData);
}

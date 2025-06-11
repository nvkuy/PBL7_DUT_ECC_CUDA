package rs_java;

public abstract class RS_Native {

    static {
        try {
            System.loadLibrary("rs_cuda");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public native void init(int maxActiveEncode, int maxActiveDecode);

    public native void fin(int maxActiveEncode, int maxActiveDecode);

    public native void encode(int ticketId, int[] p);

    public native void decode(int ticketId, int[] x, int[] y);

    public abstract void encodeProcessAfter(int ticketId, int[] y);

    public abstract void decodeProcessAfter(int ticketId, int[] p);

}

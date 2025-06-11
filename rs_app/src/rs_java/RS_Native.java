package rs_java;

public abstract class RS_Native {

    static {
        try {

            // CUDA
//            System.loadLibrary("rs_cuda");

            // ISPC (avx2 or avx512)
            System.loadLibrary("rs_ispc_avx2");
//            System.loadLibrary("rs_ispc_avx512");

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

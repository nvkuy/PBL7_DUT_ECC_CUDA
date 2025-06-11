package rs_java;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

public class RS_Code extends RS_Native {

    // create only 1 instance of this class

    private static final int MAX_ACTIVE_ENCODE = 1024;
    private static final int MAX_ACTIVE_DECODE = 1024;

    private final FileHandler fileHandler;
    private final BlockingQueue<Integer> encodeSlot, decodeSlot;
    private final TicketData[] encodeTicketData, decodeTicketData;

    public RS_Code(FileHandler fileHandler) {

        this.fileHandler = fileHandler;

        encodeSlot = new ArrayBlockingQueue<>(MAX_ACTIVE_ENCODE);
        decodeSlot = new ArrayBlockingQueue<>(MAX_ACTIVE_DECODE);

        encodeTicketData = new TicketData[MAX_ACTIVE_ENCODE];
        decodeTicketData = new TicketData[MAX_ACTIVE_DECODE];

        for (int i = 0; i < MAX_ACTIVE_ENCODE; i++) encodeSlot.add(i);
        for (int i = 0; i < MAX_ACTIVE_DECODE; i++) decodeSlot.add(i);

        init(MAX_ACTIVE_ENCODE, MAX_ACTIVE_DECODE);

    }

    public void finish() {
        fin(MAX_ACTIVE_ENCODE, MAX_ACTIVE_DECODE);
    }

    public void encodeAsync(Object fileData, long partId, int[] p) {
        Thread.ofVirtual().start(() -> {
            int ticketId = 0;
            try {
                ticketId = encodeSlot.take();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
            encodeTicketData[ticketId] = new TicketData(fileData, partId);
            super.encode(ticketId, p);
        });
    }

    public void decodeAsync(Object fileData, long partId, int[] x, int[] y) {
        Thread.ofVirtual().start(() -> {
            int ticketId = 0;
            try {
                ticketId = decodeSlot.take();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
            decodeTicketData[ticketId] = new TicketData(fileData, partId);
            super.decode(ticketId, x, y);
        });
    }

    @Override
    public void encodeProcessAfter(int ticketId, int[] y) {
        Thread.ofVirtual().start(() -> {
            TicketData ticketData = encodeTicketData[ticketId];
            try {
                encodeSlot.put(ticketId);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
            fileHandler.encodeProcessAfter(
                    (FileHandler.FileEncodeData) ticketData.fileData(), ticketData.partId(), y);
        });
    }

    @Override
    public void decodeProcessAfter(int ticketId, int[] p) {
        Thread.ofVirtual().start(() -> {
            TicketData ticketData = decodeTicketData[ticketId];
            try {
                decodeSlot.put(ticketId);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
            fileHandler.decodeProcessAfter(
                    (FileHandler.FileDecodeData) ticketData.fileData(), ticketData.partId(), p);
        });
    }

    private record TicketData(Object fileData, long partId) {}

}

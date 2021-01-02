package utils;

import java.util.List;

public class SequenceUtils {
    public static <T> void swap(T[] data, int i, int j) {
        T tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }

    public static <T> void swap(boolean[] data, int i, int j) {
        boolean tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }

    public static void swap(char[] data, int i, int j) {
        char tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }

    public static void swap(byte[] data, int i, int j) {
        byte tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }

    public static void swap(int[] data, int i, int j) {
        int tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }

    public static <T> void swap(List<T> data, int i, int j) {
        T tmp = data.get(i);
        data.set(i, data.get(j));
        data.set(j, tmp);
    }

    public static void swap(long[] data, int i, int j) {
        long tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }

    public static void swap(double[] data, int i, int j) {
        double tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}

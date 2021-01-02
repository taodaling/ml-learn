package numeric;

import utils.ArrayIndex;

import java.util.Arrays;

public class Matrix {
    private double[] data;
    private int[] shape;
    private ArrayIndex ai;

    public Matrix(int... shape) {
        this.shape = shape;
        ai = new ArrayIndex(shape);
        data = new double[ai.totalSize()];
    }

    public void reshape(int... shape) {
        this.shape = shape;
        ai = new ArrayIndex(shape);
        if (ai.totalSize() != data.length) {
            throw new IllegalStateException();
        }
    }

    public int[] getShape() {
        return shape.clone();
    }

    public void fill(double x) {
        Arrays.fill(data, x);
    }

    public int size() {
        return data.length;
    }

    public double get(int i) {
        return ai.indexOf(i);
    }


    public double get(int i, int j) {
        return ai.indexOf(i, j);
    }

    public double get(int i, int j, int k) {
        return ai.indexOf(i, j, k);
    }

    public void set(int i, double x) {
        data[ai.indexOf(i)] = x;
    }


    public void set(int i, int j, double x) {
        data[ai.indexOf(i)] = x;
    }


    public void set(int i, int j, int k, double x) {
        data[ai.indexOf(i, j, k)] = x;
    }
}

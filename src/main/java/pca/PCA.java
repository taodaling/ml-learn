package pca;

import algo.DataNormalizer;
import numeric.Vector;

import java.util.Arrays;

public class PCA {
    double[][] u;
    double[][] ut;
    DataNormalizer normalizer;

    public PCA(double[][] u, DataNormalizer normalizer) {
        this.u = u;
        this.ut = Vector.transpose(u);
        this.normalizer = normalizer;
    }

    public double[] reverse(double[] y) {
        double[] ans = Vector.matrixMul(u, y);
        normalizer.reverse(ans);
        return ans;
    }

    public double[] compress(double[] x) {
        x = x.clone();
        normalizer.normalize(x);
        return Vector.matrixMul(ut, x);
    }

    public double error2(double[] x) {
        double[] ans = reverse(compress(x));
        return Vector.length2(Vector.minus(ans, x));
    }

    public double error2(double[][] x) {
        return Arrays.stream(x).mapToDouble(this::error2).sum();
    }

    @Override
    public String toString() {
        return "pca:" + u.length + "->" + ut.length;
    }
}

package algo;

import numeric.KahamSummation;
import utils.MathUtils;

public class DataNormalizer {
    private double[] range;
    private double[] mean;

    public DataNormalizer(double[][] x) {
        int m = x.length;
        int n = x[0].length;
        range = new double[n];
        mean = new double[n];
        for (int i = 0; i < n; i++) {
            double max = -MathUtils.INF;
            double min = -max;
            KahamSummation sum = new KahamSummation();
            for (int j = 0; j < m; j++) {
                max = Math.max(max, x[j][i]);
                min = Math.min(min, x[j][i]);
                sum.add(x[j][i]);
            }
            range[i] = max - min;
            mean[i] = sum.sum() / m;
            if (range[i] < 1e-10) {
                range[i] = 1;
            }
        }
    }

    public void normalize(double[][] x) {
        int m = x.length;
        for (int i = 0; i < m; i++) {
            normalize(x[i]);
        }
    }

    public void reverse(double[][] x) {
        int m = x.length;
        for (int i = 0; i < m; i++) {
            reverse(x[i]);
        }
    }

    public void normalize(double[] x) {
        int n = x.length;
        for (int j = 0; j < n; j++) {
            x[j] = (x[j] - mean[j]) / range[j];
        }
    }

    public void reverse(double[] x) {
        int n = x.length;
        for (int j = 0; j < n; j++) {
            x[j] = x[j] * range[j] + mean[j];
        }
    }
}

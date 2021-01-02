package kmeans;

import numeric.Vector;
import utils.Pair;

public class KMeans {
    private double[][] center;

    public KMeans(double[][] centers) {
        this.center = centers;
    }

    public Pair<int[], double[][]> group(double[][] xs) {
        int[] ans = new int[xs.length];
        double[][] asCenter = new double[xs.length][];
        double total = 0;
        for (int i = 0; i < xs.length; i++) {
            int index = -1;
            double best = 0;
            for (int j = 0; j < center.length; j++) {
                double cand = Vector.length2(Vector.minus(center[j], xs[i]));
                if (index == -1 || cand < best) {
                    index = j;
                    best = cand;
                }
            }
            total += best;
            ans[i] = index;
            asCenter[i] = center[index].clone();
        }
        return new Pair<>(ans, asCenter);
    }
}

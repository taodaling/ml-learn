package ad;

import numeric.Vector;

public class AnomalyDetectionTrainer {
    public AnomalyDetection train(double[][] x) {
        int m = x.length;
        int n = x[0].length;
        NormalDistribution[] nd = new NormalDistribution[n];
        NormalDistributionTrainer ndt = new NormalDistributionTrainer();
        for (int i = 0; i < n; i++) {
            double[] col = Vector.flat(Vector.subrect(x, 0, m - 1, i, i));
            nd[i] = ndt.train(col);
        }
        return new AnomalyDetection(nd);
    }
}

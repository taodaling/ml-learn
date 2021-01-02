package ad;

public class AnomalyDetection {
    NormalDistribution[] nd;

    public AnomalyDetection(NormalDistribution[] nd) {
        this.nd = nd;
    }

    public double apply(double[] x) {
        double ans = 1;
        for (int i = 0; i < x.length; i++) {
            ans *= nd[i].apply(x[i]);
        }
        return ans;
    }
}

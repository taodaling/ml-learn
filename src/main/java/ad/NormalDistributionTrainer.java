package ad;

import numeric.KahamSummation;

import java.util.Arrays;

public class NormalDistributionTrainer {
    public NormalDistribution train(double[] x) {
        if (x.length == 0) {
            return new NormalDistribution(0, 1);
        }
        double mu = Arrays.stream(x).average().getAsDouble();
        KahamSummation sum = new KahamSummation();
        for (double y : x) {
            sum.add((y - mu) * (y - mu));
        }
        double sigma = Math.sqrt(sum.sum() / x.length);
        return new NormalDistribution(mu, sigma);
    }
}

package logistic;

import algo.Hypothesis;

import java.util.Arrays;

public class MultiLogisticRegressionAdapter implements Hypothesis<double[]> {
    public MultiLogisticRegressionAdapter(Hypothesis<Double>... hs) {
        this.hs = hs;
    }

    Hypothesis<Double>[] hs;

    @Override
    public double[] apply(double[] x) {
        return Arrays.stream(hs).mapToDouble(h -> h.apply(x))
                .toArray();
    }
}

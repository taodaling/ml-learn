package algo;

import lombok.extern.slf4j.Slf4j;
import numeric.Vector;
import utils.Pair;

import java.util.function.Function;
import java.util.function.ToDoubleFunction;

@Slf4j
public class GradientDescending {
    private ToDoubleFunction<double[]> mf;
    private Function<double[], double[]> g;

    public GradientDescending(ToDoubleFunction<double[]> mf,
                              Function<double[], double[]> g) {
        this.mf = mf;
        this.g = g;
    }

    public Pair<double[], Double> optimize(double[] theta, int maxRound,
                                           double learnRate) {
        GradientLearnRateController controller = new GradientLearnRateController(learnRate, mf.applyAsDouble(theta),
                maxRound);
        while (controller.hasNext()) {
            double[] gradient = g.apply(theta);
            double[] cand;
            cand = Vector.plus(theta, Vector.mul(gradient, -controller.getLearnRate()));
            theta = cand;
            controller.next(mf.applyAsDouble(cand));
            log.info("{}", controller);
        }
        return new Pair<>(theta, controller.getCost());
    }

}

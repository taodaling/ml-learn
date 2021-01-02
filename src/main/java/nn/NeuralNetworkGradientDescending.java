package nn;

import algo.GradientLearnRateController;
import lombok.extern.slf4j.Slf4j;
import numeric.Vector;
import utils.Pair;

import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;

import static numeric.Vector.randomInit;

@Slf4j
public class NeuralNetworkGradientDescending {
    private ToDoubleFunction<double[][][]> mf;
    private Function<double[][][], double[][][]> g;
    private Random random;
    private int[] shape;
    private AtomicBoolean status = new AtomicBoolean();

    public void stop() {
        status.compareAndSet(true, false);
    }

    public NeuralNetworkGradientDescending(ToDoubleFunction<double[][][]> mf,
                                           Function<double[][][], double[][][]> g, Random random,
                                           int[] shape) {
        this.mf = mf;
        this.g = g;
        this.random = random;
        this.shape = shape;
    }

    public Pair<double[][][], Double> optimize(int maxRound,
                                               double learnRate) {
        return optimize(maxRound, learnRate, x -> {
        });
    }

    public Pair<double[][][], Double> optimize(int maxRound,
                                               double learnRate,
                                               Consumer<double[][][]> callback) {
        status.set(true);
        double[][][] theta = NeuralNetworkUtils.createTheta(shape);
        randomInit(theta, -1, 1, random);

        GradientLearnRateController controller = new GradientLearnRateController(learnRate, mf.applyAsDouble(theta),
                maxRound);
        while (status.get() && controller.hasNext()) {
            double[][][] gradient = g.apply(theta);
            double[][][] cand;
            cand = Vector.plus(theta, Vector.mul(gradient, -controller.getLearnRate()));
            controller.next(mf.applyAsDouble(cand));
            theta = cand;
            log.info("report: {}", controller);
            callback.accept(theta);
        }
        return new Pair<>(theta, controller.getCost());
    }

}

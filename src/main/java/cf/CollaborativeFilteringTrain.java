package cf;

import algo.GradientLearnRateController;
import lombok.extern.slf4j.Slf4j;
import numeric.Vector;
import utils.Pair;

import java.util.Random;

@Slf4j
public class CollaborativeFilteringTrain {
    Random random = new Random();

    public Pair<CollaborativeFiltering, Double> train(double[][] rated, double[][] rating, double lambda, int numFeature,
                                                      double speed, int roundLimit) {
        int numU = rated.length;
        int numG = rating[0].length;
        double[] avg = new double[numG];
        for (int i = 0; i < numG; i++) {
            double sum = 0;
            int cnt = 0;
            for (int j = 0; j < numU; j++) {
                sum += rating[j][i] * rated[j][i];
                cnt += Math.round(rated[j][i]);
            }
            avg[i] = cnt == 0 ? 1 : sum / cnt;
        }
        rating = Vector.deepClone(rating);
        for (int i = 0; i < numG; i++) {
            for (int j = 0; j < numU; j++) {
                if (rated[j][i] > 1e-8) {
                    rating[j][i] -= avg[i];
                }
            }
        }
        double[][] x = new double[numG][numFeature];
        double[][] theta = new double[numU][numFeature];
        Vector.randomInit(x, -0.3, 0.3, random);
        Vector.randomInit(theta, -0.3, 0.3, random);
        CollaborativeFilteringGradientFunction gf = new CollaborativeFilteringGradientFunction(rating, rated, lambda);
        CollaborativeFilteringCostFunction cf = new CollaborativeFilteringCostFunction(rating, rated, lambda);
        GradientLearnRateController rateController = new GradientLearnRateController(speed, cf.apply(theta, x), roundLimit);

        while (rateController.hasNext()) {
            log.info("{}", rateController);
            Pair<double[][], double[][]> g = gf.apply(theta, x);
            double rate = rateController.getLearnRate();
            x = Vector.subtract(x, Vector.mul(g.b, rate));
            theta = Vector.subtract(theta, Vector.mul(g.a, rate));
            rateController.next(cf.apply(theta, x));
        }

        return new Pair<>(new CollaborativeFiltering(x, theta, avg), rateController.getCost());
    }


}

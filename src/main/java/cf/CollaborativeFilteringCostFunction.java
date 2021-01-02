package cf;

import numeric.Vector;

import java.util.Arrays;
import java.util.stream.Stream;

public class CollaborativeFilteringCostFunction {
    private double[][] rating;
    private double[][] rated;
    private double lambda;

    public CollaborativeFilteringCostFunction(double[][] rating, double[][] rated, double lambda) {
        this.rating = rating;
        this.rated = rated;
        this.lambda = lambda;
    }

    public double apply(double[][] theta, double[][] x) {
        int numU = rated.length;
        int numG = rated[0].length;
        double sum = 0;
        for (int i = 0; i < numU; i++) {
            for (int j = 0; j < numG; j++) {
                double estimate = Vector.dotmul(theta[i], x[j]);
                sum += rated[i][j] * (estimate - rating[i][j]) * (estimate - rating[i][j]);
            }
        }
        sum += lambda * Stream.of(theta, x).flatMap(t -> Arrays.stream(t))
                .mapToDouble(t -> Vector.dotmul(t, t)).sum();
        return sum / 2;
    }
}

package cf;

import numeric.Vector;
import utils.Pair;

public class CollaborativeFilteringGradientFunction {
    private double[][] rating;
    private double[][] rated;
    private double lambda;

    public CollaborativeFilteringGradientFunction(double[][] rating, double[][] rated, double lambda) {
        this.rating = rating;
        this.rated = rated;
        this.lambda = lambda;
    }

    public Pair<double[][], double[][]> apply(double[][] theta, double[][] x) {
        int numU = rated.length;
        int numG = rated[0].length;
        int feature = theta[0].length;
        double[][] dTheta = Vector.mul(theta, lambda);
        double[][] dX = Vector.mul(x, lambda);
        for (int i = 0; i < numU; i++) {
            for (int j = 0; j < numG; j++) {
                double estimate = Vector.dotmul(theta[i], x[j]);
                double diff = (estimate - rating[i][j]) * rated[i][j];
                for (int k = 0; k < feature; k++) {
                    dTheta[i][k] += diff * x[j][k];
                    dX[j][k] += diff * theta[i][k];
                }
            }
        }
        return new Pair<>(dTheta, dX);
    }
}

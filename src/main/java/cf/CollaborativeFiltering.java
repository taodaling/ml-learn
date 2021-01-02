package cf;

import numeric.Vector;

public class CollaborativeFiltering {
    private double[][] x;
    private double[][] theta;
    private double[] avg;

    public CollaborativeFiltering(double[][] x, double[][] theta, double[] avg) {
        this.x = x;
        this.theta = theta;
        this.avg = avg;
    }

    public double apply(int u, int g) {
        return Vector.dotmul(x[g], theta[u]) + avg[g];
    }
}

package nn;

import algo.Sigmoid;
import numeric.KahamSummation;
import numeric.Vector;
import utils.Pair;

public class NeuralNetworkUtils {
    public static double[][][] createTheta(int... levels){
        double[][][] theta = new double[levels.length - 1][][];
        for (int i = 0; i < levels.length - 1; i++) {
            theta[i] = new double[levels[i + 1]][levels[i]];
        }
        return theta;
    }

    public static Pair<double[][], double[][]> forwardPropagation(double[] xs, double[][][] theta) {
        double[][] as = new double[theta.length + 1][];
        double[][] zs = new double[theta.length + 1][];
        double[] a = xs;
        for (int i = 1; i <= theta.length; i++) {
            as[i - 1] = a;
            double[] z = Vector.matrixMul(theta[i - 1], a);
            zs[i] = z;
            for (int j = 0; j < z.length; j++) {
                z[j] = Sigmoid.apply(z[j]);
            }
            if (i < theta.length) {
                z[0] = 1;
            }
            a = z;
        }
        as[theta.length] = a;
        return new Pair<>(as, zs);
    }

}

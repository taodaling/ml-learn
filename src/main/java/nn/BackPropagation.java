package nn;

import utils.Pair;

import java.util.function.Function;

import static numeric.Vector.*;

public class BackPropagation implements Function<double[][][], double[][][]> {
    private double[][] xs;
    private double[][] ys;
    private double regularLambda;
    private int[] shape;

    public BackPropagation(double[][] xs, double[][] ys, double regularLambda, int[] shape) {
        this.xs = xs;
        this.ys = ys;
        this.regularLambda = regularLambda;
        this.shape = shape;
    }

    private static double[] mulCommonItem(double[] a, double[] b) {
        double[] ans = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            ans[i] = a[i] * b[i] * (1 - b[i]);
        }
        return ans;
    }

    @Override
    public double[][][] apply(double[][][] theta) {
        int m = xs.length;
        int level = theta.length;
        double[][][] Deltas = NeuralNetworkUtils.createTheta(shape);

        for (int t = 0; t < xs.length; t++) {
            Pair<double[][], double[][]> res = NeuralNetworkUtils.forwardPropagation(xs[t], theta);
            double[][] a = res.a;
            double[][] z = res.b;
            double[][] delta = new double[level + 1][];
            delta[level] = minus(a[level], ys[t]);
            for (int l = level - 1; l >= 0; l--) {
                double[] common = mulCommonItem(delta[l + 1], a[l + 1]);
                delta[l] = matrixMul(common, theta[l]);
                for(int i = 0; i < common.length; i++){
                    for(int j = 0; j < shape[l]; j++){
                        Deltas[l][i][j] += common[i] * a[l][j];
                    }
                }
            }
        }

        for (int l = 0; l < Deltas.length; l++) {
            for (int i = 0; i < Deltas[l].length; i++) {
                for (int j = 0; j < Deltas[l][i].length; j++) {
                    if (j != 0) {
                        Deltas[l][i][j] += regularLambda * theta[l][i][j];
                    }
                    Deltas[l][i][j] /= m;
                }
            }
        }
        return Deltas;
    }
}

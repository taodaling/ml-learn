package pca;

import algo.DataNormalizer;
import numeric.Vector;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import utils.Pair;

import static numeric.Vector.*;

public class PCATrainer {
    public Pair<PCA, Double> train(double[][] xs){
        return train(xs, -1);
    }
    public Pair<PCA, Double> train(double[][] xs, int k) {
        xs = Vector.deepClone(xs);
        int m = xs.length;
        int n = xs[0].length;
        double[][] sum = new double[n][n];
        DataNormalizer dn = new DataNormalizer(xs);
        dn.normalize(xs);

        for (double[] x : xs) {
            double[][] mat = matrixMul(asColumnVector(x), asRowVector(x));
            sum = plus(mat, sum);
        }

        double[][] sigma = mul(sum, 1d / m);
        RealMatrix rm = new Array2DRowRealMatrix(sigma);
        SingularValueDecomposition svd = new SingularValueDecomposition(rm);
        RealMatrix diag = svd.getS();
        double top = 0;
        double bot = diag.getTrace();
        double credit;
        if (k == -1) {
            k = n;
            credit = 0;
            if (bot != 0) {
                for (int j = 0; j < n; j++) {
                    top += diag.getEntry(j, j);
                    credit = top / bot;
                    if (credit >= 0.99) {
                        k = j + 1;
                        break;
                    }
                }
            }
        } else {
            for (int j = 0; j < k; j++) {
                top += diag.getEntry(j, j);
            }
            credit = top / bot;
        }

        RealMatrix u = svd.getU();
        PCA pca = new PCA(u.getSubMatrix(0, n - 1, 0, k - 1).getData(),
                dn);
        return new Pair<>(pca, credit);
    }
}

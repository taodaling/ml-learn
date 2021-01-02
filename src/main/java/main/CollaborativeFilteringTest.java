package main;

import cf.CollaborativeFiltering;
import cf.CollaborativeFilteringTrain;
import numeric.Vector;
import utils.MatUtils;
import utils.Pair;

public class CollaborativeFilteringTest {
    public static void main(String[] args) {
        String filename = "D:\\Temp\\ex8_movies.mat";
        double[][] Y = Vector.transpose(MatUtils.load2d(filename, "Y"));
        double[][] R = Vector.transpose(MatUtils.load2dUint8(filename, "R"));
        int feature = 50;
        double lambda = 1;
        Pair<CollaborativeFiltering, Double> res = new CollaborativeFilteringTrain().train(R, Y, lambda, feature, 1e-3, (int) 1e3);
        StringBuilder ans = new StringBuilder();
        for (int i = 0; i < Y.length; i++) {
            for (int j = 0; j < Y[0].length; j++) {
                if (R[i][j] == 0) {
                    ans.append('?');
                } else {
                    ans.append(Y[i][j]);
                }
                ans.append(":").append(res.a.apply(i, j)).append(' ');
            }
            ans.append(System.lineSeparator());
        }
        System.out.println(ans);
    }
}

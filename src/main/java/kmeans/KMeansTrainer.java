package kmeans;

import numeric.Vector;
import utils.MathUtils;
import utils.Pair;
import utils.Tuple3;

import java.util.Arrays;
import java.util.Random;

public class KMeansTrainer {
    Random random = new Random();

    public Pair<double[][], Double> train(double[][] x, int k) {
        k = Math.min(x.length, k);
        int m = x.length;
        int n = x[0].length;
        int[] prev = new int[m];
        Arrays.fill(prev, -1);
        int[] next = new int[m];
        double[][] center = new double[k][];
        for (int i = 0; i < k; i++) {
            center[i] = x[random.nextInt(m)].clone();
        }

        int[] size = new int[k];
    //    double prevTotal;
        double total = MathUtils.INF;
       // double lastTotal;
        while (true) {
            //regroup
          //  prevTotal = total;
            total = 0;
           // lastTotal = 0;
            int differ = 0;
            Arrays.fill(size, 0);
            for (int i = 0; i < m; i++) {
                int index = -1;
                double best = 0;
                for (int j = 0; j < k; j++) {
                    double dist2 = Vector.length2(Vector.minus(x[i], center[j]));
                    if (index == -1 || best > dist2) {
                        index = j;
                        best = dist2;
                    }
//                    if (prev[i] == j) {
//                        lastTotal += dist2;
//                    }
                }
                next[i] = index;
                if (prev[i] != next[i]) {
                    differ++;
                }
                total += best;
                size[next[i]]++;
            }
//            if(lastTotal > prevTotal + 1e-4){
//                throw new RuntimeException();
//            }

            for (int i = 0; i < k; i++) {
                Arrays.fill(center[i], 0);
            }
            for (int i = 0; i < m; i++) {
                int to = next[i];
                for (int j = 0; j < n; j++) {
                    center[to][j] += x[i][j];
                }
            }
            for (int i = 0; i < k; i++) {
                if (size[i] == 0) {
                    //reinit the center
                    center[i] = x[random.nextInt(m)].clone();
                    continue;
                }
                for (int j = 0; j < n; j++) {
                    center[i][j] /= size[i];
                }
            }
            int[] tmp = prev;
            prev = next;
            next = tmp;
            if (differ == 0) {
                break;
            }
        }

        return new Pair<>(center, total / m);
    }
}

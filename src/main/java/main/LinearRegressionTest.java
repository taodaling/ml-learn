package main;

import algo.DataNormalizer;
import algo.GradientDescending;
import linear.LinearRegression;
import linear.LinearRegressionCostFunction;
import linear.LinearRegressionGradientFunction;
import lombok.extern.slf4j.Slf4j;
import numeric.Vector;
import utils.MatUtils;
import utils.Pair;

import java.util.ArrayList;
import java.util.List;

@Slf4j
public class LinearRegressionTest {
    public static void main(String[] args) {
        test();
    }

    public static double[][] formatInput(double[][] x) {
        double[] col = Vector.flat(x);
        double[] last = col;
        for (int i = 0; i < 8; i++) {
            last = Vector.mul(col, last);
            x = Vector.hstack(
                    Vector.asColumnVector(last), x);
        }
        x = Vector.hstack(Vector.asColumnVector(Vector.constant(x.length, 1)), x);
        return x;
    }

    public static void test() {
        String filename = "D:/ex5data1.mat";
        double[][] trainX = MatUtils.load2d(filename, "X");
        double[] trainY = MatUtils.load1d(filename, "y");

        double[][] validateX = MatUtils.load2d(filename, "Xval");
        double[] validateY = MatUtils.load1d(filename, "yval");

        double[][] testX = MatUtils.load2d(filename, "Xtest");
        double[] testY = MatUtils.load1d(filename, "ytest");

        log.info("train set size is {}", trainY.length);
        log.info("valid set size is {}", validateY.length);
        log.info("test set size is {}", testY.length);

        trainX = formatInput(trainX);
        validateX = formatInput(validateX);
        testX = formatInput(testX);
        DataNormalizer dn = new DataNormalizer(trainX);
        dn.normalize(trainX);
        dn.normalize(validateX);
        dn.normalize(testX);

        List<Pair<double[], Double>> cand = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            log.info("train the {}-th model", i + 1);
            double lambda = 1e-3 * Math.pow(3, i);
            LinearRegressionCostFunction mf = new LinearRegressionCostFunction(trainX, trainY, lambda);
            LinearRegressionGradientFunction gf = new LinearRegressionGradientFunction(trainX, trainY, lambda);
            GradientDescending gd = new GradientDescending(mf, gf);
            cand.add(gd.optimize(new double[trainX[0].length], (int) 1e6, 1));
        }
        double[] bestChoice = null;
        double bestCost = Double.MAX_VALUE;
        double trainCost = -1;
        for (Pair<double[], Double> lr : cand) {
            double lambda = 0;
            LinearRegressionCostFunction mf = new LinearRegressionCostFunction(validateX, validateY, lambda);
            double cost = mf.applyAsDouble(lr.a);
            if (cost < bestCost) {
                bestChoice = lr.a;
                bestCost = cost;
                trainCost = lr.b;
            }
        }

        LinearRegressionCostFunction mf = new LinearRegressionCostFunction(testX, testY, 0);
        double cost = mf.applyAsDouble(bestChoice);

        log.info("The picked model is {}", new LinearRegression(bestChoice));
        log.info("The cost in train set is {}", trainCost);
        log.info("The cost in valid set is {}", bestCost);
        log.info("The cost in test set is {}", cost);

        LinearRegression lr = new LinearRegression(bestChoice);
        for (int i = 0; i < testX.length; i++) {
            log.info("expect {}, actual {}", testY[i], lr.apply(testX[i]));
        }
    }
}

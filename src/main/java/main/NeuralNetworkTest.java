package main;

import algo.*;
import logistic.*;
import lombok.extern.slf4j.Slf4j;
import nn.*;
import utils.LogisticRegressionAccuracy;
import utils.Pair;

import java.util.Random;
@Slf4j
public class NeuralNetworkTest {

    static Random random = new Random();

    public static Pair<double[][], double[]> gen(int n) {
        double[][] xs = new double[n][3];
        double[] ys = new double[n];
        for (int i = 0; i < n; i++) {
            xs[i][0] = 1;
            xs[i][1] = random.nextDouble() * 200 - 100;
            xs[i][2] = random.nextDouble() * 200 - 100;
            //  xs[i][3] = xs[i][1] * xs[i][1];
            //  xs[i][4] = xs[i][2] * xs[i][2];
            // xs[i][5] = xs[i][1] * xs[i][2];
            if ((xs[i][1] - 100) * (xs[i][1] - 100) + (xs[i][2] + 100) * (xs[i][2] + 100) <= 900 ||
                    (xs[i][1]) * (xs[i][1] ) + (xs[i][2]) * (xs[i][2]) <= 900) {
                ys[i] = 1;
            } else {
            }
        }
        return new Pair<>(xs, ys);
    }

    public static Pair<double[][], double[][]> gen2(int n) {
        double[][] xs = new double[n][3];
        double[][] ys = new double[n][2];
        for (int i = 0; i < n; i++) {
            xs[i][0] = 1;
            xs[i][1] = random.nextDouble() * 100 - 50;
            xs[i][2] = random.nextDouble() * 100 - 50;
            //  xs[i][3] = xs[i][1] * xs[i][1];
            //  xs[i][4] = xs[i][2] * xs[i][2];
            // xs[i][5] = xs[i][1] * xs[i][2];
            if ((xs[i][1] - 30) * (xs[i][1] - 30) + (xs[i][2] + 30) * (xs[i][2] + 30) <= 1600) {
                ys[i][1] = 1;
            } else {
                ys[i][0] = 1;
            }
        }
        return new Pair<>(xs, ys);
    }

    public static void testNeuralNetwork() {
        Pair<double[][], double[][]> trainset = gen2(1000);
        Pair<double[][], double[][]> testset = gen2(100000);
        DataNormalizer dn = new DataNormalizer(trainset.a);
        dn.normalize(trainset.a);
        dn.normalize(testset.a);

        int[] shape = new int[]{trainset.a[0].length, 3, trainset.b[0].length};

        double lambda = 1e-10;
        NeuralNetworkCostFunction cf = new NeuralNetworkCostFunction(trainset.a, trainset.b, lambda);
        BackPropagation gf = new BackPropagation(trainset.a, trainset.b, lambda, shape);
        NeuralNetworkGradientBF bf = new NeuralNetworkGradientBF(cf);
        //NeuralNetworkGradientBF.test(bf, gf, 1000, 1e-2, shape);

        NeuralNetworkGradientDescending trainer = new NeuralNetworkGradientDescending(cf, gf, random, shape);
        Pair<double[][][], Double> res = trainer.optimize(1000000, 1, theta -> {
            NeuralNetwork lr = new NeuralNetwork(theta);
            log.info("acuracy is " + LogisticRegressionAccuracy.testMulti(
                    new MultiLogisticRegressionPredicate(lr),
                    trainset.a, trainset.b
            ) * 100 + "%");
        });
        NeuralNetwork lr = new NeuralNetwork(res.a);

        System.out.println("trained model is " + lr);
        System.out.println("cost is " + res.b);
        System.out.println("acuracy is " + LogisticRegressionAccuracy.testMulti(
                new MultiLogisticRegressionPredicate(lr),
                testset.a, testset.b
        ));
    }

    public static void testGradient() {
        Pair<double[][], double[]> trainset = gen(1000);
        Pair<double[][], double[]> testset = gen(100000);
        DataNormalizer dn = new DataNormalizer(trainset.a);
        dn.normalize(trainset.a);
        dn.normalize(testset.a);
        double lambda = 1e-10;
        LogisticCostFunction cf = new LogisticCostFunction(trainset.a, trainset.b, lambda);
        LogisticGradientFunction gf = new LogisticGradientFunction(trainset.a, trainset.b, lambda);
        GradientBF bf = new GradientBF(cf);
        //GradientBF.test(bf, gf, 1000, 1e-2, trainset.x[0].length);

        GradientDescending trainer = new GradientDescending(cf, gf);
        Pair<double[], Double> res = trainer.optimize(new double[trainset.a[0].length], 100000, 1e-2);
        LogisticRegression lr = new LogisticRegression(res.a);

        System.out.println("trained model is " + lr);
        System.out.println("cost is " + res.b);
        System.out.println("acuracy is " + LogisticRegressionAccuracy.testBinary(
                new LogisticRegressionPredict(lr),
                testset.a, testset.b
        ));
    }

    public static void main(String[] args) {
        testNeuralNetwork();
    }
}
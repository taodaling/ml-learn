package main;

import algo.*;
import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import logistic.*;
import lombok.extern.slf4j.Slf4j;
import nn.*;
import numeric.Vector;
import utils.LogisticRegressionAccuracy;
import utils.Pair;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

import static numeric.Vector.*;

@Slf4j
public class TrainHandWritingRecognize {
    static Random random = new Random();

    public static void main(String[] args) throws IOException {
        testByNeuralNetwork();
    }

    public static void testByLogisticRegression() throws IOException {
        double[][] X = loadMat("X");
        double[][] Y = loadMat("y");
        System.out.println(Arrays.deepToString(X));
        System.out.println(Arrays.deepToString(Y));
        double[][] xs = Vector.hstack(Vector.reshape(Vector.constant(X.length, 1), X.length, 1), X);
        double[][] classify = new double[xs.length][10];
        for (int i = 0; i < xs.length; i++) {
            for (int j = 0; j < 10; j++) {
                classify[i][(int) Math.round(Y[i][0]) % 10] = 1;
            }
        }
        LogisticRegression[] reg = new LogisticRegression[10];
        for (int v = 0; v < 10; v++) {
            if (v != 3) {
                continue;
            }
            int used = 5000;
            double[] ys = new double[Y.length];
            for (int i = 0; i < Y.length; i++) {
                ys[i] = (int) Math.round(Y[i][0]) % 10 == v ? 1 : 0;
            }
            double lambda = 0.1;
            LogisticCostFunction cf = new LogisticCostFunction(head(xs, used), head(ys, used), lambda);
            LogisticGradientFunction gf = new LogisticGradientFunction(head(xs, used), head(ys, used), lambda);
            //GradientBF.test(bf, gf, 1000, 1e-2, trainset.x[0].length);

            GradientDescending trainer = new GradientDescending(cf, gf);
            Pair<double[], Double> res = trainer.optimize(new double[xs[0].length], 500, 10);
            LogisticRegression lr = new LogisticRegression(res.a);
            reg[v] = lr;
            System.out.println("acuracy is " + LogisticRegressionAccuracy.testBinary(
                    new LogisticRegressionPredict(lr),
                    xs, ys
            ) * 100 + "%");
            System.out.println("Trained model is " + lr);

        }
//        Hypothesis<double[]> hypothesis = new MultiLogisticRegressionAdapter(reg);
//        System.out.println("acuracy is " + LogisticRegressionAccuracy.testMulti(
//                new MultiLogisticRegressionPredicate(hypothesis),
//                xs, classify
//        ));
    }

    public static int[] randomPerm(int n) {
        int[] ans = IntStream.range(0, n).toArray();
        for (int i = n - 1; i >= 0; i--) {
            int j = random.nextInt(i + 1);
            int tmp = ans[i];
            ans[i] = ans[j];
            ans[j] = tmp;
        }
        return ans;
    }

    public static void shuffle(double[][] x, int[] perm) {
        double[][] y = x.clone();
        for (int i = 0; i < perm.length; i++) {
            x[perm[i]] = y[i];
        }
    }

    public static void testByNeuralNetwork() throws IOException {
        double[][] X = loadMat("X");
        double[][] Y = loadMat("y");
        System.out.println(Arrays.deepToString(X));
        System.out.println(Arrays.deepToString(Y));
        double[][] xs = Vector.hstack(Vector.reshape(Vector.constant(X.length, 1), X.length, 1), X);
        double[][] ys = new double[Y.length][10];
        for (int i = 0; i < Y.length; i++) {
            int which = (int) Math.round(Y[i][0]) % 10;
            ys[i][which] = 1;
        }
//        int[] perm = randomPerm(X.length);
//        shuffle(xs, perm);
//        shuffle(ys, perm);
        int head = (int) (xs.length * 1);
        int tail = xs.length;
        Pair<double[][], double[][]> trainset = new Pair<>(head(xs, head), head(ys, head));
        Pair<double[][], double[][]> testset = new Pair<>(tail(xs, tail), tail(xs, tail));
        double lambda = 1;

        int[] shape = new int[]{trainset.a[0].length, 10, 10, trainset.b[0].length};
        NeuralNetworkCostFunction cf = new NeuralNetworkCostFunction(trainset.a, trainset.b, lambda);
        BackPropagation gf = new BackPropagation(trainset.a, trainset.b, lambda, shape);
        NeuralNetworkGradientBF bf = new NeuralNetworkGradientBF(cf);

        //NeuralNetworkGradientBF.test(bf, gf, 1000, 1e-2, shape);
        NeuralNetworkGradientDescending trainer = new NeuralNetworkGradientDescending(cf, gf, random, shape);
        new Thread() {
            @Override
            public void run() {
                super.run();
                try {
                    System.in.read();
                } catch (IOException e) {
                    throw new RuntimeException();
                }
                trainer.stop();
            }
        }.start();

        Pair<double[][][], Double> res = trainer.optimize(1000000, 1, theta -> {
            NeuralNetwork lr = new NeuralNetwork(theta);
            log.info("acuracy is " + LogisticRegressionAccuracy.testMulti(
                    new MultiLogisticRegressionPredicate(lr),
                    testset.a, testset.b
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

    public static double[][] loadMat(String s) throws IOException {
        File file = new File("D:\\ex4data1.mat");
        MatFileReader read = new MatFileReader(file);
        MLArray mlArray = read.getMLArray(s);//mat存储的就是img矩阵变量的内容
        MLDouble d = (MLDouble) mlArray;
        double[][] matrix = (d.getArray());//只有jmatio v0.2版本中才有d.getArray方法
        return matrix;
    }
}

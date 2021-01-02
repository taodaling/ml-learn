package main;

import ad.AnomalyDetection;
import ad.AnomalyDetectionTrainer;
import lombok.extern.slf4j.Slf4j;
import numeric.Vector;
import org.math.plot.Plot2DPanel;
import utils.MatUtils;

import javax.swing.*;
import java.util.Arrays;

public class AnomalyDetectionTest {

    static double[][] fix(double[][] mat) {
//        double[] x = Vector.col(mat, 0);
//        double[] y = Vector.col(mat, 1);
//        double[] xy = Vector.mul(x, y);
//        double[] xPy = Vector.plus(x, y);
//        double[] xSy = Vector.plus(x, y);
//        return Vector.hstack(Vector.asColumnVector(xy),
//                Vector.asColumnVector(xPy),
//                Vector.asColumnVector(xSy),
//                mat);
        return mat;
    }

    public static void main(String[] args) {
        String fn = "D:\\Temp\\ex8data1.mat";
        double[][] trainSet = fix(MatUtils.load2d(fn, "X"));
        double[][] Xval = fix(MatUtils.load2d(fn, "Xval"));
        double[] yval = MatUtils.load1d(fn, "yval");
//        Xval = trainSet;
//        yval = Vector.constant(yval.length, 0);
        System.out.println("trainset.length=" + trainSet.length);
        System.out.println("Xval.length=" + Xval.length);
        System.out.println("yval=" + Arrays.toString(yval));
        System.out.println("\\sum yval=" + Arrays.stream(yval).sum());
        if (Xval.length != yval.length) {
            throw new RuntimeException();
        }
        Plot2DPanel plot = new Plot2DPanel();
        // add a line plot to the PlotPanel
        plot.addScatterPlot("my plot", trainSet);
        // put the PlotPanel in a JFrame, as a JPanel
        JFrame frame = new JFrame("a plot panel");
        frame.setContentPane(plot);
        frame.setSize(800, 800);
        frame.setVisible(true);


        AnomalyDetection ad = new AnomalyDetectionTrainer().train(trainSet);
        double epsilon = 1e-10;
        double diff = 0;
        double correct = 0;
        for (int i = 0; i < Xval.length; i++) {
            double prob = ad.apply(Xval[i]);
            double contrib = prob - (1 - yval[i]);
            diff += contrib * contrib;
            if ((prob < epsilon) == (yval[i] == 1)) {
                correct++;
            }
        }
        System.out.println("训练总误差:" + diff);
        System.out.println("正确率:" + (correct / Xval.length));

    }
}

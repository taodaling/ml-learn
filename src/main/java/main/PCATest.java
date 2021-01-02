package main;

import org.math.plot.Plot2DPanel;
import pca.PCA;
import pca.PCATrainer;
import utils.MatUtils;
import utils.Pair;

import javax.swing.*;
import java.awt.*;
import java.util.Arrays;

public class PCATest {
    public static void main(String[] args) {

    }

    public static void test2(){

    }

    public static void test1(){
        double[][] xs = MatUtils.load2d("D:\\Temp\\ex7data1.mat", "X");
//        mat = new double[][]{
//                {0, 0},
//                {1, 1},
//                {0, 1},
//                {1, 0}
//        };
        System.out.println(xs.length + "," + xs[0].length);
        Pair<PCA, Double> pca = new PCATrainer().train(xs, 1);
        System.out.println(pca);
        System.out.println(pca.a.error2(xs));

        // add a line plot to the PlotPanel
        Plot2DPanel plot = new Plot2DPanel();
        plot.addScatterPlot("my plot", Color.BLUE, xs);
        // put the PlotPanel in a JFrame, as a JPanel
        JFrame frame = new JFrame("a plot panel");
        frame.setContentPane(plot);
        frame.setSize(800, 800);
        frame.setVisible(true);

        double[][] casted = Arrays.stream(xs).map(x -> pca.a.reverse(pca.a.compress(x))).toArray(i -> new double[i][]);
        plot.addScatterPlot("my plot", Color.GREEN, casted);
    }
}

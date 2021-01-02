package main;

import kmeans.KMeans;
import kmeans.KMeansTrainer;
import lombok.extern.slf4j.Slf4j;
import numeric.Vector;
import org.math.plot.Plot2DPanel;
import utils.ImageUtils;
import utils.MatUtils;
import utils.Pair;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;

@Slf4j
public class KMeansTest {
    public static void main(String[] args) {
        compressImage();
    }

    public static void compressImage() {
        int[][] img = ImageUtils.loadImage("D:\\Temp\\bird_small.png");
        double[][][] mat = ImageUtils.castToARGB(img);
        double[][] xs = Vector.reshape(Vector.flat(mat), mat[0].length * mat.length, mat[0][0].length);
        System.out.println(img.length + ", " + img[0].length);
//        xs = new double[][]{
//                {0, 0},
//                {1, 1},
//                {2, 0},
//                {0, 2},
//                {2, 2}
//        };

//        // create your PlotPanel (you can use it as a JPanel)
//        Plot2DPanel plot = new Plot2DPanel();
//
//        // add a line plot to the PlotPanel
//        plot.addScatterPlot("my plot", xs);
//        // put the PlotPanel in a JFrame, as a JPanel
//        JFrame frame = new JFrame("a plot panel");
//        frame.setContentPane(plot);
//        frame.setSize(800, 800);
//        frame.setVisible(true);

        int k = 100;
        Pair<double[][], Double> res = null;
        log.info("Start");
        KMeansTrainer trainer = new KMeansTrainer();
        for (int i = 0; i < 10; i++) {
            Pair<double[][], Double> cand = trainer.train(xs, k);
            log.info("Run res{}", cand);
            if (res == null || res.b > cand.b) {
                res = cand;
            }
        }

        Pair<int[], double[][]> group = new KMeans(res.a).group(xs);
//        // plot.removeAll();
//        List<double[]>[] split = new List[k];
//        for (int i = 0; i < k; i++) {
//            split[i] = new ArrayList<>();
//        }
//        for (int i = 0; i < xs.length; i++) {
//            int g = group.a[i];
//            split[g].add(xs[i]);
//        }
//        Color[] colors = new Color[k];
//        Random random = new Random(0);
//        for (int i = 0; i < k; i++) {
//            colors[i] = new Color(random.nextFloat(), random.nextFloat(), random.nextFloat());
//        }
//        for (int i = 0; i < k; i++) {
//            plot.addScatterPlot("group-" + i, colors[i], split[i].toArray(new double[0][]));
//            //plot.addScatterPlot("group-center-" + i, ccolors[i], new double[][]{res.a[i]});
//        }
        double[][][] compressed = new double[mat.length][mat[0].length][];
        for (int i = 0; i < mat.length; i++) {
            for(int j = 0; j < mat[0].length; j++){
                int index = group.a[i * mat[0].length + j];
                compressed[i][j] = group.b[i * mat[0].length + j];
            }
        }
        int[][] backImg = ImageUtils.castToARGB(compressed);
        ImageUtils.write(backImg, "D:\\Temp\\bird_small_2.png");
    }

    public static void group() {
        double[][] xs = MatUtils.load2d("D:\\Temp\\ex7data2.mat", "X");
        System.out.println(xs.length + ", " + xs[0].length);
//        xs = new double[][]{
//                {0, 0},
//                {1, 1},
//                {2, 0},
//                {0, 2},
//                {2, 2}
//        };

        // create your PlotPanel (you can use it as a JPanel)
        Plot2DPanel plot = new Plot2DPanel();

        // add a line plot to the PlotPanel
        plot.addScatterPlot("my plot", xs);
        // put the PlotPanel in a JFrame, as a JPanel
        JFrame frame = new JFrame("a plot panel");
        frame.setContentPane(plot);
        frame.setSize(800, 800);
        frame.setVisible(true);

        int k = 3;
        Pair<double[][], Double> res = null;
        log.info("Start");
        KMeansTrainer trainer = new KMeansTrainer();
        for (int i = 0; i < 10; i++) {
            Pair<double[][], Double> cand = trainer.train(xs, k);
            log.info("Run res{}", cand);
            if (res == null || res.b > cand.b) {
                res = cand;
            }
        }

        Pair<int[], double[][]> group = new KMeans(res.a).group(xs);
        // plot.removeAll();
        List<double[]>[] split = new List[k];
        for (int i = 0; i < k; i++) {
            split[i] = new ArrayList<>();
        }
        for (int i = 0; i < xs.length; i++) {
            int g = group.a[i];
            split[g].add(xs[i]);
        }
        Color[] colors = new Color[]{
                Color.GREEN, Color.BLUE, Color.RED
        };
        Color[] ccolors = new Color[]{
                Color.darkGray, Color.CYAN, Color.YELLOW
        };
        for (int i = 0; i < k; i++) {
            plot.addScatterPlot("group-" + i, colors[i], split[i].toArray(new double[0][]));
            //plot.addScatterPlot("group-center-" + i, ccolors[i], new double[][]{res.a[i]});
        }
    }
}

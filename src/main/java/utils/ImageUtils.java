package utils;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;

import static java.awt.image.BufferedImage.TYPE_3BYTE_BGR;
import static java.awt.image.BufferedImage.TYPE_4BYTE_ABGR;

public class ImageUtils {
    public static int[][] loadImage(String filename) {
        BufferedImage img = null;
        try {
            img = ImageIO.read(new File(filename));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        int[][] mat = new int[img.getWidth()][img.getHeight()];
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[i].length; j++) {
                mat[i][j] = img.getRGB(i, j);
            }
        }
        return mat;
    }

    public static void write(int[][] mat, String filename) {
        int n = mat.length;
        int m = mat[0].length;
        BufferedImage img = new BufferedImage(m, n, TYPE_4BYTE_ABGR);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                img.setRGB(i, j, mat[i][j]);
            }
        }
        try {
            ImageIO.write(img, "png", new File(filename));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public static int[][] castToARGB(double[][][] mat) {
        int n = mat.length;
        int m = mat[0].length;
        int[][] ans = new int[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                for (int k = 0; k < 4; k++) {
                    ans[i][j] |= (int) Math.round(mat[i][j][k]) << (8 * k);
                }
            }
        }
        return ans;
    }

    public static double[][][] castToARGB(int[][] img) {
        int n = img.length;
        int m = img[0].length;
        double[][][] ans = new double[n][m][4];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                for (int k = 0; k < 4; k++) {
                    ans[i][j][k] = (img[i][j] >> (k * 8)) & 255;
                }
            }
        }
        return ans;
    }
}

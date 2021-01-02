package utils;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLUInt8;
import numeric.Vector;

import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;

public class MatUtils {
    public static double[][] load2d(String filename, String s) {
        File file = new File(filename);
        MatFileReader read = null;
        try {
            read = new MatFileReader(file);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        MLArray mlArray = read.getMLArray(s);//mat存储的就是img矩阵变量的内容
        MLDouble d = (MLDouble) mlArray;
        double[][] matrix = (d.getArray());//只有jmatio v0.2版本中才有d.getArray方法
        return matrix;
    }

    public static double[][] load2dUint8(String filename, String s) {
        File file = new File(filename);
        MatFileReader read = null;
        try {
            read = new MatFileReader(file);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        MLArray mlArray = read.getMLArray(s);//mat存储的就是img矩阵变量的内容
        MLUInt8 d = (MLUInt8) mlArray;
        byte[][] matrix = (d.getArray());//只有jmatio v0.2版本中才有d.getArray方法
        double[][] casted = new double[matrix.length][matrix[0].length];
        for(int i = 0; i < matrix.length; i++){
            for(int j = 0; j < matrix[i].length; j++){
                casted[i][j] = matrix[i][j];
            }
        }
        return casted;
    }

    public static double[] load1d(String filename, String s) {
        return Vector.flat(load2d(filename, s));
    }
}

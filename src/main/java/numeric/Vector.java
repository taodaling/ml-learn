package numeric;

import java.util.Arrays;
import java.util.Random;

public class Vector {
    public static double[] matrixMul(double[][] a, double[] b) {
        return flat(matrixMul(a, asColumnVector(b)));
    }

    public static double[] matrixMul(double[] a, double[][] b) {
        return flat(matrixMul(asRowVector(a), b));
    }

    public static double[][][] randomInit(double[][][] mat, double l, double r, Random random) {
        for (int i = 0; i < mat.length; i++) {
            randomInit(mat[i], l, r, random);
        }
        return mat;
    }

    public static double[][] randomInit(double[][] mat, double l, double r, Random random) {
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[i].length; j++) {
                mat[i][j] = random.nextDouble() * (r - l) + l;
            }
        }
        return mat;
    }

    public static double[][] matrixMul(double[][] a, double[][] b) {
        assert a[0].length == b.length;
        int n = a.length;
        int mid = b.length;
        int m = b[0].length;
        double[][] ans = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                for (int k = 0; k < mid; k++) {
                    ans[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return ans;
    }

    public static double[][] asColumnVector(double[] v) {
        return reshape(v, v.length, 1);
    }

    public static double[][] asRowVector(double[] v) {
        return reshape(v, 1, v.length);
    }

    public static double[][] transpose(double[][] mat) {
        int n = mat.length;
        int m = mat[0].length;
        double[][] ans = new double[m][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                ans[j][i] = mat[i][j];
            }
        }
        return ans;
    }

    public static byte[][] transpose(byte[][] mat) {
        int n = mat.length;
        int m = mat[0].length;
        byte[][] ans = new byte[m][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                ans[j][i] = mat[i][j];
            }
        }
        return ans;
    }

    public static double[][] transpose(double[] mat) {
        return transpose(asColumnVector(mat));
    }

    public static double dotmul(double[] a, double[] b) {
        assert a.length == b.length;
        int n = a.length;
        KahamSummation sum = new KahamSummation();
        for (int i = 0; i < n; i++) {
            sum.add(a[i] * b[i]);
        }
        return sum.sum();
    }

    public static double[] minus(double[] a, double[] b) {
        assert a.length == b.length;
        int n = a.length;
        double[] ans = new double[n];
        for (int i = 0; i < n; i++) {
            ans[i] = a[i] - b[i];
        }
        return ans;
    }

    public static <T> T deepClone(T x) {
        if (!x.getClass().isArray()) {
            throw new IllegalArgumentException();
        }
        if (x instanceof double[]) {
            return (T) ((double[]) x).clone();
        }
        Object[] casted = (Object[]) x;
        Object[] clone = casted.clone();
        for (int i = 0; i < clone.length; i++) {
            clone[i] = deepClone(clone[i]);
        }
        return (T) clone;
    }

    public static <T> void deepFill(T x, double v) {
        if (!x.getClass().isArray()) {
            throw new IllegalArgumentException();
        }
        if (x instanceof double[]) {
            double[] casted = (double[]) x;
            for (int i = 0; i < casted.length; i++) {
                casted[i] = v;
            }
            return;
        }
        Object[] casted = (Object[]) x;
        for (int i = 0; i < casted.length; i++) {
            deepFill(casted[i], v);
        }
    }

    public static double[][] reshape(double[] seq, int n, int m) {
        if (seq.length != n * m) {
            throw new IllegalArgumentException();
        }
        double[][] ans = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                ans[i][j] = seq[i * m + j];
            }
        }
        return ans;
    }

    public static double[][][] reshape(double[] seq, int n, int m, int k) {
        if (seq.length != n * m * k) {
            throw new IllegalArgumentException();
        }
        double[][][] ans = new double[n][m][k];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                for (int t = 0; t < k; t++) {
                    ans[i][j][t] = seq[(i * m + j) * k + t];
                }
            }
        }
        return ans;
    }

    public static double[][] reshape(double[][] mat, int n, int m) {
        return reshape(flat(mat), n, m);
    }

    public static double[] constant(int n, double init) {
        double[] ans = new double[n];
        Arrays.fill(ans, init);
        return ans;
    }

    public static double[] flat(double[][] mat) {
        double[] ans = new double[mat.length * mat[0].length];
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[0].length; j++) {
                ans[i * mat[0].length + j] = mat[i][j];
            }
        }
        return ans;
    }

    public static double[] flat(double[][][] mat) {
        double[] ans = new double[mat.length * mat[0].length * mat[0][0].length];
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[0].length; j++) {
                for (int k = 0; k < mat[0][0].length; k++) {
                    ans[(i * mat[0].length + j) * mat[0][0].length + k] = mat[i][j][k];
                }
            }
        }
        return ans;
    }

    public static double[][] vstack(double[][]... arr) {
        int sumRow = 0;
        for (double[][] x : arr) {
            sumRow += x.length;
        }
        double[][] ans = new double[sumRow][arr[0][0].length];
        int offset = 0;
        for (double[][] xs : arr) {
            for (int i = 0; i < xs.length; i++) {
                for (int j = 0; j < xs[i].length; j++) {
                    ans[i + offset][j] = xs[i][j];
                }
            }
            offset += xs.length;
        }
        return ans;
    }

    public static void mulEq(double[] seq, double x) {
        for (int i = 0; i < seq.length; i++) {
            seq[i] *= x;
        }
    }

    public static double[] mul(double[] seq, double x) {
        seq = seq.clone();
        mulEq(seq, x);
        return seq;
    }

    public static double[][] mul(double[][] seq, double x) {
        double[][] ans = new double[seq.length][];
        for (int i = 0; i < seq.length; i++) {
            ans[i] = mul(seq[i], x);
        }
        return ans;
    }

    public static double[][][] mul(double[][][] seq, double x) {
        double[][][] ans = new double[seq.length][][];
        for (int i = 0; i < seq.length; i++) {
            ans[i] = mul(seq[i], x);
        }
        return ans;
    }

    public static double[] mul(double[] a, double[] b) {
        assert a.length == b.length;
        double[] ans = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            ans[i] = a[i] * b[i];
        }
        return ans;
    }

    public static double[] mul(double[] a, double[] b, double[] c) {
        assert a.length == b.length;
        double[] ans = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            ans[i] = a[i] * b[i] * c[i];
        }
        return ans;
    }

    public static double[] plus(double[] a, double[] b) {
        double[] ans = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            ans[i] = a[i] + b[i];
        }
        return ans;
    }

    public static double[][] plus(double[][] a, double[][] b) {
        assert a.length == b.length;
        double[][] ans = new double[a.length][];
        for (int i = 0; i < a.length; i++) {
            ans[i] = plus(a[i], b[i]);
        }
        return ans;
    }

    public static double[][][] plus(double[][][] a, double[][][] b) {
        assert a.length == b.length;
        double[][][] ans = new double[a.length][][];
        for (int i = 0; i < a.length; i++) {
            ans[i] = plus(a[i], b[i]);
        }
        return ans;
    }


    public static double[] subtract(double[] a, double[] b) {
        double[] ans = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            ans[i] = a[i] - b[i];
        }
        return ans;
    }

    public static double[][] subtract(double[][] a, double[][] b) {
        double[][] ans = new double[a.length][];
        for (int i = 0; i < a.length; i++) {
            ans[i] = subtract(a[i], b[i]);
        }
        return ans;
    }

    public static double[][] hstack(double[][]... arr) {
        int sumCol = 0;
        for (double[][] x : arr) {
            sumCol += x[0].length;
        }
        double[][] ans = new double[arr[0].length][sumCol];
        int offset = 0;
        for (double[][] xs : arr) {
            for (int i = 0; i < xs.length; i++) {
                for (int j = 0; j < xs[i].length; j++) {
                    ans[i][j + offset] = xs[i][j];
                }
            }
            offset += xs[0].length;
        }
        return ans;
    }

    public static double[] head(double[] x, int n) {
        return Arrays.copyOf(x, n);
    }

    public static double[][] head(double[][] x, int n) {
        return Arrays.copyOf(x, n);
    }

    public static double[] tail(double[] x, int n) {
        return Arrays.copyOfRange(x, x.length - n, x.length);
    }

    public static double[][] tail(double[][] x, int n) {
        return Arrays.copyOfRange(x, x.length - n, x.length);
    }

    public static double length2(double[] x) {
        return dotmul(x, x);
    }

    public static double[][] subrect(double[][] mat, int b, int t, int l, int r) {
        double[][] ans = new double[t - b + 1][r - l + 1];
        for (int i = b; i <= t; i++) {
            for (int j = l; j <= r; j++) {
                ans[i - b][j - l] = mat[i][j];
            }
        }
        return ans;
    }

    public static double[] row(double[][] mat, int i) {
        return mat[i].clone();
    }

    public static double[] col(double[][] mat, int i) {
        return flat(subrect(mat, 0, mat.length - 1, i, i));
    }


    public static double length(double[] x) {
        return Math.sqrt(length2(x));
    }
}

import java.util.List;

public interface IFunctionApproximator {
    void train(List<Sample> samples);
    double[] approximate(double[] inputs);
}

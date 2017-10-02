public interface INeuron {
    double[] getWeights();
    double propagate(double[] inputs);
}

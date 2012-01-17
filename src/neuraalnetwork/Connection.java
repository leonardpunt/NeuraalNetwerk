package neuraalnetwork;

public class Connection {

    private Neuron connectionFrom;
    private Weight weight;

    Connection(Neuron connectionFrom, Weight weight) {
        this.connectionFrom = connectionFrom;
        this.weight = weight;
    }

    public Neuron getConnectionFrom() {
        return connectionFrom;
    }

    public Weight getWeight() {
        return weight;
    }
}

package ai.konduit.serving.example;

import ai.konduit.serving.pipeline.CustomPipelineStepUDF;
import org.datavec.api.records.Record;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.iterator.BertIterator;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.text.tokenization.tokenizerfactory.BertWordPieceTokenizerFactory;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.List;

public class BertUDF implements CustomPipelineStepUDF, LabeledSentenceProvider{
    private final BertIterator bertIterator;
    private Record[] currentRecords;


    public BertUDF() {
        super();
        String path = System.getProperty("bertudf.vocab");
        File vocabFile = new File(path);

        try {

            BertWordPieceTokenizerFactory tokenizer = new BertWordPieceTokenizerFactory(vocabFile, true, true, StandardCharsets.UTF_8);
            bertIterator = BertIterator.builder()
                    .tokenizer(tokenizer)
                    .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, 100)
                    .minibatchSize(1)
                    .sentenceProvider(this)
                    .featureArrays(BertIterator.FeatureArrays.INDICES_MASK)
                    .vocabMap(tokenizer.getVocab())
                    .task(BertIterator.Task.SEQ_CLASSIFICATION)
                    .build();
        } catch (IOException e) {
            throw new IllegalStateException("Vocabulary file missing.  Check that JVM property -Dbertudf.vocab is set porperly.", e);
        }
    }


    @Override
    public Record[] udf(Record[] records) {
        if (records == null) {
            throw new NullPointerException("records can not be null.");
        }

        currentRecords = records;
        MultiDataSet dataSet = bertIterator.next();
        NDArrayWritable writable = new NDArrayWritable(dataSet.getFeatures()[0]);
        org.datavec.api.records.impl.Record ret = new org.datavec.api.records.impl.Record(Collections.singletonList(writable), null);
        return new Record[] { ret };
    }

    //region LabeledSentenceProvider Methods

    @Override
    public boolean hasNext() {
        return currentRecords != null;
    }

    @Override
    public Pair<String, String> nextSentence() {
        List<Writable> writables = currentRecords[0].getRecord();
        return Pair.of(writables.get(0).toString(), "");
    }

    @Override
    public void reset() {

    }

    @Override
    public int totalNumSentences() {
        return 0;
    }

    @Override
    public List<String> allLabels() {
        return Collections.singletonList("");
    }

    @Override
    public int numLabelClasses() {
        return 1;
    }
    //endregion
}

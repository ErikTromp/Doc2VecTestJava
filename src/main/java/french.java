import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.documentiterator.SimpleLabelAwareIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

public class french {
	public static void main(String[] args) {
		try {
			BufferedReader br = new BufferedReader(
					   new InputStreamReader(new FileInputStream("fr_FR.txt"), "UTF8"));
			List<LabelledDocument> docs = new ArrayList<LabelledDocument>();
			String line = br.readLine();
			Long offset = 0L;
			while (line != null) {
				LabelledDocument doc = new LabelledDocument();
				doc.setContent(line);
				doc.addLabel("SENT" + offset);
				docs.add(doc);
				
				offset = offset + 1;
				line = br.readLine();
			}
			br.close();
			TokenizerFactory tokenizer  = new DefaultTokenizerFactory();
			SimpleLabelAwareIterator iterator = new SimpleLabelAwareIterator(docs);
			ParagraphVectors parVecs = new ParagraphVectors.Builder()
				.minWordFrequency(10)
				.iterations(15)
				.epochs(2)
				.layerSize(300)
				.learningRate(0.025)
				.windowSize(5)
				.batchSize(1500)
				.iterate(iterator)
				.trainWordVectors(true)
				.sampling(0.0)
				.tokenizerFactory(tokenizer).build();
			parVecs.fit();
			WordVectorSerializer.writeParagraphVectors(parVecs, "fr_doc_vectors.bin");
		} catch (Exception e) {
			
		}
	}
}
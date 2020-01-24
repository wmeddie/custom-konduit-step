package ai.konduit.serving.example;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.datavec.api.records.impl.Record;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Text;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Collections;

import static org.junit.Assert.*;

/**
 * Unit test for simple App.
 */
public class AppTest {

    @BeforeClass
    public static void beforeTest() throws IOException {
        InputStream resource = AppTest.class.getResourceAsStream("vocab.txt");
        File tempFile = File.createTempFile(String.valueOf(resource.hashCode()), ".tmp");
        tempFile.deleteOnExit();
        FileUtils.copyInputStreamToFile(resource, tempFile);
        System.setProperty("bertudf.vocab", tempFile.getAbsolutePath());
    }

    @Test
    public void shouldAnswerWithTrue() {
        assertTrue(true);
    }

    @Test
    public void testUdfWithNullRecords() {
        try {

            new BertUDF().udf(null);
            fail("Should thow NPE.");
        } catch (NullPointerException e){
            assertTrue(true);
        }
    }

    @Test
    public void testUdfWithSentence() {
        String input = "i have a pen i have pineapple";
        Text inputText = new Text(input);
        Record record = new Record(Collections.singletonList(inputText), null);
        org.datavec.api.records.Record[] outputs = new BertUDF().udf(new org.datavec.api.records.Record[]{record});
        assertNotNull(outputs);
        assertTrue(outputs.length > 0);
        assertTrue(outputs[0].getRecord().get(0) instanceof NDArrayWritable);

        NDArrayWritable writable = (NDArrayWritable)outputs[0].getRecord().get(0);
        assertEquals(2, writable.get().rank());
        assertEquals(1, writable.get().shape()[0]);
        assertEquals(100, writable.get().shape()[1]);
    }
}

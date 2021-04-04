import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

class gameTest {

    @BeforeEach
    void setUp() {
        System.out.println("Test Start!!!");
    }

    @AfterEach
    void tearDown() {
        System.out.println("Test End!!!");
    }

    @Test
    void jarge() throws IOException {
        game g = new game();
        assertEquals("Player2 Win!!!", g.jarge(1, 2));
        assertEquals("Player2 Win!!!", g.jarge(2, 3));
        assertEquals("Player1 Win!!!", g.jarge(1, 3));
        assertEquals("Player1 Win!!!", g.jarge(2, 1));
        assertEquals("Draw!!!", g.jarge(1, 1));
        assertEquals("Draw!!!", g.jarge(2, 2));
        assertEquals("Draw!!!", g.jarge(3, 3));
    }

    @Test
    public void NullException() throws NullPointerException{
        Exception exception = assertThrows(NullPointerException.class, () -> {
            new game().jarge(null,1);
        });
    }
}

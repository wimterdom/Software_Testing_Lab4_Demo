/**
 * 
 */
import java.io.*;
/**
 * @author Administrator
 *
 */
public class game {
	public static String jarge(Integer x, Integer y) throws IOException {
		if (x == y) {
			return "Draw!!!";
		}else if (x==1 && y==2) {
			return "Player2 Win!!!";
		}else if (x==1 && y==3) {
			return "Player1 Win!!!";
		}else if (x==2 && y==1) {
			return "Player1 Win!!!";
		}else if (x==2 && y==3) {
			return "Player2 Win!!!";
		}else if (x==3 && y==1) {
			return "Player2 Win!!!";
		}else if(x==3 && y==2) {
			return "Player1 Win!!!";
		}else {
			return "Input Error!!!";
		}	
	}
}

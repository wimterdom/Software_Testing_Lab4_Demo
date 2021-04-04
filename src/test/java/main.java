import java.util.*;
import java.io.*;

class china_girl {

    public static void main (String []args) throws IOException{
        int a,b;
        Scanner sc = new Scanner(System.in);
        System.out.println("=== Welcome to Rock-Paper-Scissors game ===");
        System.out.println("Enter Player 1 choice(1:Rock, 2:Paper, 3:Scissors)");
        a=sc.nextInt();
        System.out.println("Enter Player 2 choice(1:Rock, 2:Paper, 3:Scissors)");
        b=sc.nextInt();
        System.out.println(game.jarge(a,b));
    }
}
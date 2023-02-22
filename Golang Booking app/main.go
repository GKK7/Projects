package main

//import packages
import (
	"fmt"
	"sync"
	"time"
)

const conferenceTickets uint = 50

var conferenceName = "GO Conference"
var remainingTickets uint = 50

var bookings = make([]UserData, 0)

type UserData struct {
	firstName       string
	familyname      string
	email           string
	NumberofTickets uint
}

// waitgroup to execute the current thread
var wg = sync.WaitGroup{}

func main() {

	greetUsers()

	for {

		firstName, familyname, email, userTickets := getuserinput()
		isValidName, isValidEmail, isValidTicketNumber := validateUserInput(firstName, familyname, email, uint(userTickets))
		// if there are not sufficient available tickets then break out of the loop immediately
		if isValidName && isValidEmail && isValidTicketNumber && userTickets <= remainingTickets {

			bookTicket(userTickets, firstName, familyname, email)

			wg.Add(1)
			go sendTicket(userTickets, firstName, familyname, email)

			//call function print first names
			firstNames := getFirstNames()
			fmt.Printf("The first names of the bookings are: %v\n", firstNames)

			// check for remaining tickets
			if remainingTickets == 0 {
				//end program logic
				fmt.Println("Conference is fully booked now")
				break
			}
		} else {
			if !isValidName {
				fmt.Printf("Invalid name - too short\n")
			}
			if !isValidEmail {
				fmt.Printf("Invalid email\n")
			}
			if !isValidTicketNumber {
				fmt.Printf("Not enough remaining tickets\n")
			}
		}
	}
	wg.Wait()
}

func sendTicket(userTickets uint, firstName string, familyname string, email string) {
	time.Sleep(3 * time.Second)
	var ticket = fmt.Sprintf("%v tickets for %v %v", userTickets, firstName, familyname)
	fmt.Println("#########")
	fmt.Printf("Sending ticket: \n %v to email address %v \n", ticket, email)
	fmt.Println("#########")
	wg.Done()
}

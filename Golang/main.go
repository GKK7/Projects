package main

//import formatting package
import (
	"fmt"
	"sync"
	"time"
)

const conferenceTickets uint = 50

var conferenceName = "GO Conference"
var remainingTickets uint = 50

// make bookings a dynamic map by including 0 as it will expand
var bookings = make([]UserData, 0)

type UserData struct {
	firstName       string
	familyname      string
	email           string
	NumberofTickets uint
}

// waitgroup to execute the current thread
var wg = sync.WaitGroup{}

// everything goes in the function
// Println - print newline
func main() {
	//go implicitly figures out the datatype based on data - string/integer
	//%T - type of data

	greetUsers()

	// either with an empty array {} or filled up with values
	// fixed size array in this case 50

	//if go cant figure out the data type, it needs to be told
	//go identifies problems as they are written

	for {

		// isValidCity := city == "Sofia" || city == "Varna"
		// !isValidCity
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

func greetUsers() {
	fmt.Printf("Welcome to %v booking app\n", conferenceName)
	fmt.Printf("We have a total of %v tickets and there are %v tickets still available\n", conferenceTickets, remainingTickets)
	fmt.Println("Get your tickets to attend")
}

func getFirstNames() []string {
	firstNames := []string{}
	for _, booking := range bookings {
		firstNames = append(firstNames, booking.firstName)
	}
	return firstNames
}

func getuserinput() (string, string, string, uint) {
	var firstName string
	var familyname string
	var email string
	var userTickets uint

	//ask users for input
	fmt.Println("Enter your first name:")
	fmt.Scan(&firstName)
	fmt.Println("Enter your family name")
	fmt.Scan(&familyname)
	fmt.Println("Enter your email:")
	fmt.Scan(&email)
	fmt.Println("How many tickets would you like to book?")
	fmt.Scan(&userTickets)

	return firstName, familyname, email, userTickets
}

func bookTicket(userTickets uint, firstName string, familyname string, email string) {
	remainingTickets = remainingTickets - userTickets

	// create a map for each user

	var userData = UserData{
		firstName:       firstName,
		familyname:      familyname,
		email:           email,
		NumberofTickets: userTickets,
	}
	//make bookings a type of map

	bookings = append(bookings, userData)
	fmt.Printf("List of bookings is %v\n", bookings)

	//output everything
	fmt.Printf("The whole slice: %v\n", bookings)
	fmt.Printf("The first ticket is sold to %v\n", bookings[0])
	fmt.Printf("Slice length is %v\n", len(bookings))
	fmt.Printf("%v %v booked %v tickets for the %v email\n", firstName, familyname, userTickets, email)
	fmt.Printf("Thank you %v %v for booking %v at %v you will get you confirmation at %v\n", firstName, familyname, userTickets, email, email)
	fmt.Printf("There are now %v tickets available\n", remainingTickets)
}

func sendTicket(userTickets uint, firstName string, familyname string, email string) {
	time.Sleep(3 * time.Second)
	var ticket = fmt.Sprintf("%v tickets for %v %v", userTickets, firstName, familyname)
	fmt.Println("#########")
	fmt.Printf("Sending ticket: \n %v to email address %v \n", ticket, email)
	fmt.Println("#########")
	wg.Done()
}

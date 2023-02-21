package main

import (
	"fmt"
	"strings"
)

func validateUserInput(firstName string, familyname string, email string, userTickets uint) (bool, bool, bool) {
	isValidName := len(firstName) >= 2 && len(familyname) >= 2
	isValidEmail := strings.Contains(email, "@")
	isValidTicketNumber := userTickets > 0 && userTickets <= remainingTickets
	return isValidName, isValidEmail, isValidTicketNumber
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
	//make bookings a map

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

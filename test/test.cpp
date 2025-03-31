#include<stdio.h>
#include<string.h>
#define _CRT_SECURE_NO_WARNINGS
int main(void)
{
	char firstname[50], lastname[50];
	printf("print your first name:");
	scanf("%s", firstname);

	printf("print your last name:");
	scanf("%s", lastname);
	int firstlength = strlen(firstname);
	 int lastlength = strlen(lastname);

	printf("%s%s\n", firstname, lastname);
	// printf("%*d %*d\n", firstlength,lastlength);
    printf("%*d %*d\n", 5, firstlength, 5, lastlength);
	printf("%s%s\n", firstname, lastname);
	printf("%-*d%-*d\n", 5, firstlength, 5, lastlength);



	return 0;
}
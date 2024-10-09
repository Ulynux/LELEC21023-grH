#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/*char * b64_encode(char *bytes) {
	char* output = malloc(strlen(bytes)+sizeof(bytes[0]));
	//This functions only copies the bytes into a new array
	//TODO : Implement encoding here
	memcpy(output,bytes,strlen(bytes)+sizeof(bytes[0]));
	return output;
}*/
const char b64_table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
char * b64_encode(char *bytes) {

	int len = strlen(bytes);
	int padding = 0;
	if (len % 3 != 0 ){
		padding = (3 - (len % 3)) %3;
	} 
	int output_length = 4*strlen( len + padding + 1);
	
	char* output = malloc(output_length);
	if (output == NULL) {
		return NULL;
	}
	int i, j;
    uint32_t buffer;

    // Iterate over input bytes in chunks of 3 bytes
    for (i = 0, j = 0; i < len;) {
        buffer = 0;
        
        // Read up to 3 bytes into a 24-bit buffer
        buffer |= (i < len ? (unsigned char)bytes[i++] : 0) << 16;
        buffer |= (i < len ? (unsigned char)bytes[i++] : 0) << 8;
        buffer |= (i < len ? (unsigned char)bytes[i++] : 0);

        // Encode 24 bits as 4 Base64 characters (6 bits each)
        output[j++] = b64_table[(buffer >> 18) & 0x3F];
        output[j++] = b64_table[(buffer >> 12) & 0x3F];
        output[j++] = b64_table[(buffer >> 6) & 0x3F];
        output[j++] = b64_table[buffer & 0x3F];
    }

    // Apply padding if needed
    if (padding == 1) {
        output[--j] = '=';
    } else if (padding == 2) {
        output[--j] = '=';
        output[--j] = '=';
    }

    // Null-terminate the output string
    output[output_length] = '\0';

    return output;
}

int main(int argc, char *argv[])
{
	if (argc>=2) {
		char* encoded_bytes;
		encoded_bytes = b64_encode(argv[1]);
		printf(encoded_bytes);
		printf("\n");
	}
	return EXIT_SUCCESS;
}

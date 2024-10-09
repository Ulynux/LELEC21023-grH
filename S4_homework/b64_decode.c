#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// Base64 decoding table
const char b64_table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// Function to map Base64 characters to their respective values
int b64_value(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;  // Invalid character
}

// Function to decode a Base64-encoded string
char *b64_decode(const char *encoding) {
    int len = strlen(encoding);
    int padding = 0;

    // Check how many padding characters ('=') are at the end
    if (len > 0 && encoding[len - 1] == '=') padding++;
    if (len > 1 && encoding[len - 2] == '=') padding++;

    // Allocate memory for the decoded output (always 3 bytes for every 4 Base64 characters, minus padding)
    int output_len = (len * 3) / 4 - padding;
    char *output = malloc(output_len + 1);  // +1 for null terminator
    if (output == NULL) {
        return NULL;  // Memory allocation failed
    }

    // Decode the Base64 string
    int i, j;
    uint32_t buffer = 0;  // 24-bit buffer to store 4 Base64 values (6 bits each)

    for (i = 0, j = 0; i < len;) {
        buffer = (b64_value(encoding[i++]) << 18);
        buffer |= (b64_value(encoding[i++]) << 12);
        buffer |= (b64_value(encoding[i++]) << 6);
        buffer |= b64_value(encoding[i++]);

        if (j < output_len) output[j++] = (buffer >> 16) & 0xFF;
        if (j < output_len) output[j++] = (buffer >> 8) & 0xFF;
        if (j < output_len) output[j++] = buffer & 0xFF;
    }

    output[output_len] = '\0';

    return output;
}

int main(int argc, char *argv[]) {
    if (argc >= 2) {
        char *decoded_bytes = b64_decode(argv[1]);
        if (decoded_bytes != NULL) {
            printf("%s\n", decoded_bytes);
            free(decoded_bytes);  // Free the allocated memory
        } else {
            printf("Decoding failed.\n");
        }
    } else {
        printf("Usage: %s <base64_encoded_string>\n", argv[0]);
    }

    return EXIT_SUCCESS;
}

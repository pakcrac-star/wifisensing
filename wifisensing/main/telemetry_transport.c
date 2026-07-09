#include "telemetry_transport.h"
#include <stdio.h>

void telemetry_transport_write(const char *payload)
{
    if (payload == NULL) return;
    printf("%s\n", payload);
}
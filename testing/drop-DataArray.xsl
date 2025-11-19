<?xml version="1.0" encoding="UTF-8"?>

<!--
SPDX-FileCopyrightText: 2025 ProFACE developers

SPDX-License-Identifier: CC0-1.0
-->

<xsl:stylesheet version="1.0"
      xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
      xmlns:libxslt="http://xmlsoft.org/XSLT/"
      exclude-result-prefixes="libxslt">

    <!-- Enable pretty printing -->
    <xsl:output method="xml" indent="yes"/>
    <xsl:strip-space elements="*"/>

    <!-- Set number of spaces per indent (libxslt extension) -->
    <xsl:param name="indent-spaces" select="'2'"/>

    <!-- Identity template: copies everything unchanged -->
    <xsl:template match="@*|node()">
        <xsl:copy>
            <xsl:apply-templates select="@*|node()"/>
        </xsl:copy>
    </xsl:template>

    <!-- Keep <DataArray> but remove text content -->
    <xsl:template match="DataArray">
        <xsl:copy>
            <!-- Copy attributes -->
            <xsl:apply-templates select="@*"/>
            <!-- Copy child elements but NOT text nodes -->
            <xsl:apply-templates select="node()[not(self::text())]"/>
        </xsl:copy>
    </xsl:template>

</xsl:stylesheet>

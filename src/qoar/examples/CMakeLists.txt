build_lib_example(
  NAME qoar
  SOURCE_FILES qoar.cc
  LIBRARIES_TO_LINK
    ${libwifi}
    ${libinternet}
    ${libQoar}
    ${libinternet-apps}
)
